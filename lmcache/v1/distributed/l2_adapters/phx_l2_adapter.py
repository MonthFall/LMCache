# SPDX-License-Identifier: Apache-2.0
"""Phoenix L2 adapter for LMCache MP mode.

Asymmetric PHX: POSIX store (CPU MemoryObj → disk) + phxfs_read DMA load
(disk → device MemoryObj).  Falls back to POSIX read for CPU MemoryObj.

Configuration (JSON)::

    {"type": "phx", "base_path": "/mnt/nvme4/kv_cache/phx", "device_id": 4}
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    get_type_name_for_config,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


# ── helpers ──


def _object_key_to_filename(key: "ObjectKey") -> str:
    """Convert ObjectKey to a flat filename (same scheme as S3/HFBucket).

    The model_name may contain path separators (e.g. '/mnt/nvme4/.../model');
    replace them with '_' to produce a safe flat filename.
    """
    safe_model = key.model_name.replace('/', '_')
    base = (
        f"{safe_model}@{key.kv_rank:08x}"
        f"@{key.object_group_id:x}@{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}@{key.cache_salt}"
    return base


def _key_to_path(base_path: str, key: "ObjectKey") -> str:
    """Map an ObjectKey to a two-level directory path under *base_path*."""
    fname = _object_key_to_filename(key)
    if len(fname) >= 4:
        l1, l2, rest = fname[:2], fname[2:4], fname[4:]
        return os.path.join(base_path, l1, l2, rest)
    return os.path.join(base_path, fname)


# ── config ──


@dataclass
class PhxL2AdapterConfig(L2AdapterConfigBase):
    """Configuration for :class:`PhxL2Adapter`."""

    base_path: str
    device_id: int = 0
    buffer_size_mb: int = 2048
    use_direct_io: bool = True
    max_capacity_bytes: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PhxL2AdapterConfig":
        return cls(
            base_path=d["base_path"],
            device_id=int(d.get("device_id", d.get("cuda_gpu_id", 0))),
            buffer_size_mb=int(d.get("buffer_size_mb", 2048)),
            use_direct_io=bool(d.get("use_direct_io", True)),
            max_capacity_bytes=int(d.get("max_capacity_bytes", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": get_type_name_for_config(self),
            "base_path": self.base_path,
            "device_id": self.device_id,
            "buffer_size_mb": self.buffer_size_mb,
            "use_direct_io": self.use_direct_io,
            "max_capacity_bytes": self.max_capacity_bytes,
        }

    def help_params(self) -> list[tuple[str, str, str]]:
        return [
            ("base_path", "str", "Root directory for KV cache files"),
            ("device_id", "int", "Device ID for phxfs DMA (default: 0)"),
            ("buffer_size_mb", "int", "Device buffer size in MiB (default: 2048)"),
            ("use_direct_io", "bool", "Use O_DIRECT (default: true)"),
            ("max_capacity_bytes", "int", "Max bytes, 0=unlimited (default: 0)"),
        ]

    @classmethod
    def help(cls) -> str:
        return (
            "Phoenix L2 adapter config fields:\n"
            "- base_path (str): root directory for KV cache "
            "files (required)\n"
            "- device_id (int): Device ID for phxfs DMA "
            "(optional, default 0)\n"
            "- buffer_size_mb (int): Device buffer size in MiB "
            "(optional, default 2048)\n"
            "- use_direct_io (bool): use O_DIRECT for I/O "
            "(optional, default true)\n"
            "- max_capacity_bytes (int): max bytes to store, "
            "0=unlimited (optional, default 0)"
        )


# ── adapter ──


class PhxL2Adapter(L2AdapterInterface):
    """Phoenix KV cache L2 adapter (asymmetric PHX).

    Store:  CPU MemoryObj → POSIX write → Phoenix storage
    Load:   Phoenix storage → phxfs_read DMA → device MemoryObj (or POSIX fallback)
    Lookup: hot_cache + os.path.exists
    """

    def __init__(self, config: PhxL2AdapterConfig) -> None:
        super().__init__(max_capacity_bytes=config.max_capacity_bytes)

        self._config = config
        self._base_path = config.base_path
        os.makedirs(self._base_path, exist_ok=True)

        # ── Phoenix device DMA setup (lazy: fallback if unavailable) ──
        self._phx_cache = None
        self._phx_base_pointer: Optional[int] = None
        self._phx_buffer_size: int = config.buffer_size_mb * 1024 * 1024
        self._phx_allocator = None

        try:
            import phxcache  # type: ignore[import-untyped]
            import torch

            self._phx_cache = phxcache.PhxCache(
                device_id=config.device_id
            )
            logger.info(
                "PhxL2Adapter: PhxCache initialized (device_id=%d)",
                config.device_id,
            )

            from lmcache.v1.memory_allocators.phx_device_memory_allocator import (
                PhxDeviceMemoryAllocator,
            )

            device = f"{torch_device_type}:{config.device_id}"
            self._phx_allocator = PhxDeviceMemoryAllocator(
                self._phx_buffer_size,
                device=device,
                phx_cache=self._phx_cache,
            )
            self._phx_base_pointer = self._phx_allocator.base_pointer
            logger.info(
                "PhxL2Adapter: device buffer ready (size=%d MiB, base=0x%x)",
                config.buffer_size_mb,
                self._phx_base_pointer,
            )
        except ImportError:
            logger.warning(
                "PhxL2Adapter: phxcache not available, "
                "falling back to POSIX read"
            )
        except Exception as e:
            logger.warning(
                "PhxL2Adapter: PhxCache init failed (%s), "
                "falling back to POSIX read", e,
            )
            self._phx_cache = None

        # ── event fds ──
        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # ── task queues ──
        self._store_queue: list[tuple[L2TaskId, list, list]] = []
        self._store_lock = threading.Lock()
        self._store_completed: dict[L2TaskId, L2StoreResult] = {}
        self._store_completed_lock = threading.Lock()
        self._next_store_id = 0

        self._lookup_queue: list[tuple[L2TaskId, list]] = []
        self._lookup_lock = threading.Lock()
        self._lookup_results: dict[L2TaskId, Bitmap] = {}
        self._lookup_results_lock = threading.Lock()
        self._next_lookup_id = 0

        self._load_queue: list[tuple[L2TaskId, list, list]] = []
        self._load_lock = threading.Lock()
        self._load_results: dict[L2TaskId, Bitmap] = {}
        self._load_results_lock = threading.Lock()
        self._next_load_id = 0
        # Device-resident MemoryObjs produced by PHX DMA load (task_id ->
        # {key: obj}). Populated by _process_load when is_phx_available();
        # consumed by pop_loaded_device_objs() so the prefetch controller can
        # store them in L1 and serve retrieve via D2D (no D2H round-trip).
        self._load_device_objs: dict[L2TaskId, dict[ObjectKey, MemoryObj]] = {}

        # ── hot cache ──
        self._hot_cache: dict[str, str] = {}
        self._hot_lock = threading.Lock()

        # ── fd cache (read-only, LRU) ──
        # Avoids repeated os.open/os.close for the same cache files across
        # batch read requests.  Only used by _process_load (read path);
        # store uses .tmp files which are temporary and not worth caching.
        self._fd_cache: OrderedDict[str, int] = OrderedDict()
        self._fd_cache_lock = threading.Lock()
        self._fd_cache_max = 256

        # ── background worker ──
        self._stop_flag = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="phx-l2-worker"
        )
        self._worker_thread.start()
        logger.info("PhxL2Adapter started (base_path=%s)", self._base_path)

    # ── event fd interface ──

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ── store ──

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._store_lock:
            task_id = self._next_store_id
            self._next_store_id += 1
            self._store_queue.append((task_id, keys, objects))
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        with self._store_completed_lock:
            result = dict(self._store_completed)
            self._store_completed.clear()
            return result

    # ── lookup ──

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> L2TaskId:
        with self._lookup_lock:
            task_id = self._next_lookup_id
            self._next_lookup_id += 1
            self._lookup_queue.append((task_id, keys))
        return task_id

    def query_lookup_and_lock_result(
        self, task_id: L2TaskId
    ) -> Optional[Bitmap]:
        with self._lookup_results_lock:
            return self._lookup_results.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        pass  # no-op: file-based, no explicit lock

    # ── load ──

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._load_lock:
            task_id = self._next_load_id
            self._next_load_id += 1
            self._load_queue.append((task_id, keys, objects))
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._load_results_lock:
            return self._load_results.pop(task_id, None)

    # ── background worker ──

    def _worker_loop(self) -> None:
        while not self._stop_flag.is_set():
            busy = False

            with self._store_lock:
                store_tasks = list(self._store_queue)
                self._store_queue.clear()
            for task in store_tasks:
                self._process_store(task)
                busy = True

            with self._lookup_lock:
                lookup_tasks = list(self._lookup_queue)
                self._lookup_queue.clear()
            for task in lookup_tasks:
                self._process_lookup(task)
                busy = True

            with self._load_lock:
                load_tasks = list(self._load_queue)
                self._load_queue.clear()
            for task in load_tasks:
                self._process_load(task)
                busy = True

            if not busy:
                self._stop_flag.wait(timeout=0.01)

    def _process_store(
        self, task: tuple[L2TaskId, list[ObjectKey], list[MemoryObj]]
    ) -> None:
        """Store: CPU MemoryObj → batch write (phxfs_write_batch) or POSIX."""
        t_store_start = time.perf_counter()
        task_id, keys, objects = task
        import torch

        success_keys: list[ObjectKey] = []
        success_sizes: list[int] = []

        # ── Batch write path (phxfs_write_batch with CPU buffers) ──
        if self.is_phx_available():
            batch_reqs: list[tuple[int, int, int, int, int]] = []
            batch_keys: list[ObjectKey] = []
            batch_tmps: list[str] = []
            batch_finals: list[str] = []
            batch_sizes: list[int] = []
            batch_tensors: list[Any] = []  # keep refs alive during write
            batch_fds: list[int] = []

            for key, obj in zip(keys, objects):
                try:
                    path = _key_to_path(self._base_path, key)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    tmp_path = path + ".tmp"

                    tensor = obj.raw_tensor
                    if tensor is None:
                        continue
                    buf = tensor.contiguous()
                    size = buf.nbytes

                    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                    buf_ptr = buf.data_ptr()

                    batch_reqs.append((fd, buf_ptr, 0, size, 0))
                    batch_keys.append(key)
                    batch_tmps.append(tmp_path)
                    batch_finals.append(path)
                    batch_sizes.append(size)
                    batch_tensors.append(buf)
                    batch_fds.append(fd)
                except Exception as e:
                    logger.error("PhxL2Adapter store prep failed for %s: %s", key, e)

            if batch_reqs:
                try:
                    results = self._phx_cache.write_batch(batch_reqs)
                except Exception as e:
                    logger.error("PhxL2Adapter batch write failed: %s", e)
                    results = [-1] * len(batch_reqs)
                finally:
                    for fd in batch_fds:
                        os.close(fd)

                for i, (key, result, tmp_path, final_path, size) in enumerate(
                    zip(batch_keys, results, batch_tmps, batch_finals, batch_sizes)
                ):
                    if result == size:
                        os.rename(tmp_path, final_path)
                        with self._hot_lock:
                            self._hot_cache[_object_key_to_filename(key)] = final_path
                        success_keys.append(key)
                        success_sizes.append(size)
                    else:
                        logger.error(
                            "PhxL2Adapter batch write failed for %s: result=%d",
                            key, result)
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
        else:
            # ── Fallback: POSIX per-key write ──
            for key, obj in zip(keys, objects):
                try:
                    path = _key_to_path(self._base_path, key)
                    os.makedirs(os.path.dirname(path), exist_ok=True)

                    tensor = obj.raw_tensor
                    if tensor is None:
                        continue
                    size = tensor.nbytes

                    tmp_path = path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        if tensor.dtype == torch.bfloat16:
                            f.write(
                                tensor.contiguous().view(torch.uint8).numpy()
                            )
                        else:
                            f.write(tensor.contiguous().numpy())
                    os.rename(tmp_path, path)

                    with self._hot_lock:
                        self._hot_cache[_object_key_to_filename(key)] = path

                    success_keys.append(key)
                    success_sizes.append(size)
                except Exception as e:
                    logger.error("PhxL2Adapter store failed for %s: %s", key, e)

        self._notify_keys_stored(success_keys, success_sizes)
        total_bytes = sum(success_sizes)
        with self._store_completed_lock:
            self._store_completed[task_id] = L2StoreResult(
                success=len(success_keys) == len(keys),
                bytes_transferred=total_bytes
            )
        self._store_efd.notify()
        t_store = time.perf_counter() - t_store_start
        store_bw = (total_bytes / t_store / 1024 / 1024
                    if t_store > 0 and total_bytes > 0 else 0.0)
        logger.info(
            "PhxL2 STORE breakdown task=%d keys=%d ok=%d/%d "
            "%.1fMB in %.1fms (%.0fMB/s) mode=%s",
            task_id, len(keys), len(success_keys), len(keys),
            total_bytes / 1024 / 1024, t_store * 1000, store_bw,
            "batch" if self.is_phx_available() else "posix",
        )

    def _process_lookup(
        self, task: tuple[L2TaskId, list[ObjectKey]]
    ) -> None:
        """Lookup: check hot_cache + file existence."""
        task_id, keys = task
        bitmap = Bitmap(len(keys))
        with self._hot_lock:
            for i, key in enumerate(keys):
                fname = _object_key_to_filename(key)
                if fname in self._hot_cache:
                    bitmap.set(i)
                    continue
                path = _key_to_path(self._base_path, key)
                if os.path.exists(path):
                    self._hot_cache[fname] = path
                    bitmap.set(i)
        with self._lookup_results_lock:
            self._lookup_results[task_id] = bitmap
        self._lookup_efd.notify()

    def _process_load(
        self, task: tuple[L2TaskId, list[ObjectKey], list[MemoryObj]]
    ) -> None:
        """Load: batch PHX DMA (preferred) or per-key POSIX read.

        When PHX is available, allocate device MemoryObjs for all keys,
        open all files, and submit a single phxfs_read_batch for concurrent
        I/O.  Keys that can't be batched (file missing, pool exhausted, or
        batch read failure) fall back to per-key POSIX/DMA read.

        Instrumented with per-phase timing for performance breakdown.
        """
        t_total_start = time.perf_counter()
        task_id, keys, objects = task
        bitmap = Bitmap(len(keys))
        device_objs: dict[ObjectKey, MemoryObj] = {}

        # ── Phase 1: Resolve file paths (hot_cache + os.path.exists) ──
        t_path_start = time.perf_counter()
        key_paths: dict[int, str] = {}  # index -> path
        path_misses = 0  # keys not found on disk
        for i, (key, obj) in enumerate(zip(keys, objects)):
            with self._hot_lock:
                fname = _object_key_to_filename(key)
                path = self._hot_cache.get(fname)
            if path is None:
                path = _key_to_path(self._base_path, key)
                if not os.path.exists(path):
                    path_misses += 1
                    continue
            key_paths[i] = path
        t_path = time.perf_counter() - t_path_start

        if not key_paths:
            with self._load_results_lock:
                self._load_results[task_id] = bitmap
            self._load_efd.notify()
            logger.info(
                "PhxL2 LOAD task=%d keys=%d path_miss=%d "
                "total=%.1fms (no files found, early return)",
                task_id, len(keys), path_misses,
                (time.perf_counter() - t_total_start) * 1000,
            )
            return

        # Timing accumulators for the batch path
        t_prep = 0.0         # prep phase (alloc + fd open, interleaved)
        t_alloc = 0.0      # device obj allocation
        t_fd = 0.0          # fd open (get_read_fd)
        t_read_batch = 0.0  # phxfs_read_batch C++ call
        t_result = 0.0      # post-batch result processing
        fd_hits = 0
        fd_misses = 0
        batch_bytes = 0     # total bytes in batch read
        batch_ok_count = 0  # keys successfully read in batch
        # batch_reqs/batch_indices init'd here so the summary log can safely
        # reference len(batch_reqs) even when is_phx_available() is False.
        batch_reqs: list = []
        batch_indices: list = []

        # ── Phase 2: Batch read path (phxfs_read_batch) ──
        if self.is_phx_available():
            batch_indices: list[int] = []
            batch_reqs: list[tuple[int, int, int, int]] = []
            batch_dev_objs: list[MemoryObj] = []
            batch_fds: list[int] = []
            batch_paths: list[str] = []

            # Phase 2a: allocate device objs + open fds (interleaved)
            t_prep_start = time.perf_counter()
            for i, path in key_paths.items():
                obj = objects[i]
                cpu_tensor = obj.raw_tensor
                if cpu_tensor is None:
                    continue
                size = cpu_tensor.nbytes

                # ── allocate device MemoryObj ──
                t_a = time.perf_counter()
                device_obj = self._phx_allocator.allocate(
                    shapes=obj.metadata.shape,
                    dtypes=obj.metadata.dtype,
                    fmt=obj.metadata.fmt,
                )
                t_alloc += time.perf_counter() - t_a
                if device_obj is None:
                    continue  # pool exhausted, skip for fallback

                # ── open fd (with cache) ──
                flags = os.O_RDONLY
                if self._config.use_direct_io:
                    flags |= os.O_DIRECT
                t_f = time.perf_counter()
                # Track cache hit/miss by checking before _get_read_fd
                with self._fd_cache_lock:
                    was_cached = path in self._fd_cache
                try:
                    fd = self._get_read_fd(path, flags)
                except OSError as e:
                    self._phx_allocator.free(device_obj)
                    logger.error("PhxL2Adapter open failed for %s: %s", keys[i], e)
                    continue
                t_fd += time.perf_counter() - t_f
                if was_cached:
                    fd_hits += 1
                else:
                    fd_misses += 1

                buf_offset = device_obj.raw_tensor.data_ptr() - self._phx_base_pointer
                batch_indices.append(i)
                batch_reqs.append((fd, buf_offset, size, 0))
                batch_dev_objs.append(device_obj)
                batch_fds.append(fd)
                batch_paths.append(path)
                batch_bytes += size
            t_prep = time.perf_counter() - t_prep_start

            # Phase 2b: phxfs_read_batch (the actual DMA call, blocks until all done)
            if batch_reqs:
                t_read_start = time.perf_counter()
                try:
                    results = self._phx_cache.read_batch(
                        self._phx_base_pointer, batch_reqs)
                except Exception as e:
                    logger.error("PhxL2Adapter batch read failed: %s", e)
                    results = [-1] * len(batch_reqs)
                t_read_batch = time.perf_counter() - t_read_start

                # Phase 2c: process batch results
                t_result_start = time.perf_counter()
                for j, (idx, dev_obj, result, path) in enumerate(
                    zip(batch_indices, batch_dev_objs, results, batch_paths)
                ):
                    key = keys[idx]
                    expected = objects[idx].raw_tensor.nbytes
                    if result == expected:
                        device_objs[key] = dev_obj
                        bitmap.set(idx)
                        self._notify_keys_accessed([key])
                        batch_ok_count += 1
                    else:
                        self._phx_allocator.free(dev_obj)
                        self._invalidate_read_fd(path)
                        logger.error(
                            "PhxL2Adapter batch read failed for %s: "
                            "result=%d/%d", key, result, expected)

                # Remove successfully batched keys from fallback set
                batched_ok = {batch_indices[j]
                              for j in range(len(batch_indices))
                              if results[j] == objects[batch_indices[j]].raw_tensor.nbytes}
                for idx in batched_ok:
                    key_paths.pop(idx, None)
                t_result = time.perf_counter() - t_result_start

            # Free device objs for keys that were prepared but not batched
            # (shouldn't happen, but just in case)
            for j in range(len(batch_dev_objs)):
                if batch_indices[j] in key_paths:
                    self._phx_allocator.free(batch_dev_objs[j])

        # ── Phase 3: Fallback per-key POSIX/DMA read for remaining keys ──
        t_fb_start = time.perf_counter()
        fb_bytes = 0
        for i, path in key_paths.items():
            try:
                obj = objects[i]
                cpu_tensor = obj.raw_tensor
                if cpu_tensor is None:
                    continue
                size = cpu_tensor.nbytes

                if cpu_tensor.device.type != "cpu" and self._phx_cache is not None:
                    self._load_dma(path, cpu_tensor, size)
                else:
                    self._load_posix(path, cpu_tensor, size)

                bitmap.set(i)
                self._notify_keys_accessed([keys[i]])
                fb_bytes += size
            except Exception as e:
                logger.error("PhxL2Adapter load failed for %s: %s", keys[i], e)
        t_fallback = time.perf_counter() - t_fb_start

        with self._load_results_lock:
            self._load_results[task_id] = bitmap
            if device_objs:
                self._load_device_objs[task_id] = device_objs
        self._load_efd.notify()

        # ── Timing summary ──
        t_total = time.perf_counter() - t_total_start
        total_bytes = batch_bytes + fb_bytes
        n_batch = len(batch_reqs) if self.is_phx_available() else 0
        n_fallback = len(key_paths)
        odirect = "O_DIRECT" if self._config.use_direct_io else "buffered"
        read_bw = (batch_bytes / t_read_batch / 1024 / 1024
                   if t_read_batch > 0 and batch_bytes > 0 else 0.0)
        logger.info(
            "PhxL2 LOAD breakdown task=%d keys=%d(%s) "
            "total=%.1fms | "
            "path=%.1fms(miss=%d) | "
            "prep=%.1fms[alloc=%.1fms fd=%.1fms(hits=%d miss=%d)] | "
            "read_batch=%.1fms[%d reqs, %.1fMB, %.0fMB/s] | "
            "result=%.1fms(ok=%d) | "
            "fallback=%.1fms(%d keys, %.1fMB) | "
            "dev_objs=%d",
            task_id, len(keys), odirect,
            t_total * 1000,
            t_path * 1000, path_misses,
            t_prep * 1000,
            t_alloc * 1000, t_fd * 1000, fd_hits, fd_misses,
            t_read_batch * 1000, n_batch,
            batch_bytes / 1024 / 1024, read_bw,
            t_result * 1000, batch_ok_count,
            t_fallback * 1000, n_fallback, fb_bytes / 1024 / 1024,
            len(device_objs),
        )

    def _get_read_fd(self, path: str, flags: int) -> int:
        """Get a read fd from cache, or open a new one and cache it.

        Uses an LRU cache so that frequently-read cache files keep their fds
        open across batch read requests, avoiding repeated os.open/os.close.
        """
        with self._fd_cache_lock:
            if path in self._fd_cache:
                self._fd_cache.move_to_end(path)
                return self._fd_cache[path]
        fd = os.open(path, flags)
        with self._fd_cache_lock:
            if path in self._fd_cache:
                # Another thread opened the same path; close the duplicate
                os.close(fd)
                self._fd_cache.move_to_end(path)
                return self._fd_cache[path]
            if len(self._fd_cache) >= self._fd_cache_max:
                _, old_fd = self._fd_cache.popitem(last=False)
                os.close(old_fd)
            self._fd_cache[path] = fd
        return fd

    def _invalidate_read_fd(self, path: str) -> None:
        """Close and remove a cached fd (e.g. after a read failure)."""
        with self._fd_cache_lock:
            fd = self._fd_cache.pop(path, None)
        if fd is not None:
            os.close(fd)

    def _load_dma(self, path: str, tensor: Any, size: int) -> None:
        """Load via phxfs_read DMA to device buffer."""
        import phxcache

        dev_ptr = tensor.data_ptr()
        buf_offset = dev_ptr - self._phx_base_pointer

        flags = os.O_RDONLY
        if self._config.use_direct_io:
            flags |= os.O_DIRECT

        t0 = time.perf_counter()
        with phxcache.PhxFile(self._phx_cache, path, flags) as f:
            ret = f.read(
                buf=self._phx_base_pointer,
                buf_offset=buf_offset,
                nbyte=size,
                f_offset=0,
            )
        dt = time.perf_counter() - t0
        bw = size / dt / 1024 / 1024 if dt > 0 else 0.0
        logger.info(
            "PhxL2 LOAD_DMA single fd_path=%s size=%.1fMB "
            "%.1fms (%.0fMB/s) ret=%d",
            os.path.basename(path), size / 1024 / 1024,
            dt * 1000, bw, ret,
        )
        if ret != size:
            raise IOError(
                f"phxfs_read short read: {ret}/{size} bytes from {path}"
            )

    def _load_posix(self, path: str, tensor: Any, size: int) -> None:
        """Fallback: POSIX read into tensor."""
        import torch

        buf = tensor.contiguous()
        if buf.dtype == torch.bfloat16:
            buf = buf.view(torch.uint8)
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            f.readinto(buf.numpy())
        dt = time.perf_counter() - t0
        bw = size / dt / 1024 / 1024 if dt > 0 else 0.0
        logger.info(
            "PhxL2 LOAD_POSIX single fd_path=%s size=%.1fMB "
            "%.1fms (%.0fMB/s)",
            os.path.basename(path), size / 1024 / 1024,
            dt * 1000, bw,
        )

    def is_phx_available(self) -> bool:
        """PHX 是否可用（phx_cache 已初始化 + phx_allocator 已创建）。"""
        return self._phx_cache is not None and self._phx_allocator is not None

    def pop_loaded_device_objs(
        self, task_id: L2TaskId
    ) -> dict[ObjectKey, MemoryObj]:
        """Return device-resident MemoryObjs produced by this load task.

        Overrides the base no-op: when PHX DMA was used, returns the device
        MemoryObjs allocated from the phx pool (keyed by ObjectKey) so the
        prefetch controller can store them in L1 and serve retrieve via
        device→device scatter (D2D) instead of an H2D copy.

        Returns an empty dict for keys that fell back to POSIX read (those
        filled the controller-provided CPU obj directly) or for tasks that
        have already been popped.
        """
        with self._load_results_lock:
            return self._load_device_objs.pop(task_id, {})

    # ── lifecycle ──

    def close(self) -> None:
        self._stop_flag.set()
        self._worker_thread.join(timeout=5)

        # Close all cached read fds
        with self._fd_cache_lock:
            for fd in self._fd_cache.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fd_cache.clear()

        # Release any device objs never popped by the controller (e.g. aborted
        # prefetch requests) to avoid leaking phx pool memory.
        with self._load_results_lock:
            leaked_device_maps = list(self._load_device_objs.values())
            self._load_device_objs.clear()
        if self._phx_allocator is not None:
            for device_map in leaked_device_maps:
                for device_obj in device_map.values():
                    try:
                        self._phx_allocator.free(device_obj)
                    except Exception:
                        pass
            del self._phx_allocator
            self._phx_allocator = None
        if self._phx_cache is not None:
            self._phx_cache.close()
            self._phx_cache = None

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()
        logger.info("PhxL2Adapter closed")

    def report_status(self) -> dict:
        return {
            "is_healthy": True,
            "type": "phx",
            "base_path": self._base_path,
            "dma_enabled": self._phx_cache is not None,
            "hot_cache_size": len(self._hot_cache),
        }


# Self-register config type and adapter factory
register_l2_adapter_type("phx", PhxL2AdapterConfig)


def _create_phx_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: Optional["L1MemoryDesc"] = None,
) -> "L2AdapterInterface":
    """Create a PhxL2Adapter from config."""
    return PhxL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("phx", _create_phx_adapter)
