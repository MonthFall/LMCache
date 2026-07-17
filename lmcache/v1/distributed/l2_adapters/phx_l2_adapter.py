# SPDX-License-Identifier: Apache-2.0
"""Phoenix L2 adapter for LMCache MP mode.

Asymmetric GDS: POSIX store (CPU MemoryObj → disk) + phxfs_read DMA load
(disk → GPU MemoryObj).  Falls back to POSIX read for CPU MemoryObj.

Configuration (JSON)::

    {"type": "phx", "base_path": "/mnt/nvme4/kv_cache/phx", "cuda_gpu_id": 4}
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

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
    cuda_gpu_id: int = 0
    buffer_size_mb: int = 2048
    use_direct_io: bool = True
    max_capacity_bytes: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PhxL2AdapterConfig":
        return cls(
            base_path=d["base_path"],
            cuda_gpu_id=int(d.get("cuda_gpu_id", 0)),
            buffer_size_mb=int(d.get("buffer_size_mb", 2048)),
            use_direct_io=bool(d.get("use_direct_io", True)),
            max_capacity_bytes=int(d.get("max_capacity_bytes", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": get_type_name_for_config(self),
            "base_path": self.base_path,
            "cuda_gpu_id": self.cuda_gpu_id,
            "buffer_size_mb": self.buffer_size_mb,
            "use_direct_io": self.use_direct_io,
            "max_capacity_bytes": self.max_capacity_bytes,
        }

    def help_params(self) -> list[tuple[str, str, str]]:
        return [
            ("base_path", "str", "Root directory for KV cache files"),
            ("cuda_gpu_id", "int", "CUDA GPU ID for phxfs DMA (default: 0)"),
            ("buffer_size_mb", "int", "GPU buffer size in MiB (default: 2048)"),
            ("use_direct_io", "bool", "Use O_DIRECT (default: true)"),
            ("max_capacity_bytes", "int", "Max bytes, 0=unlimited (default: 0)"),
        ]

    @classmethod
    def help(cls) -> str:
        return (
            "Phoenix L2 adapter config fields:\n"
            "- base_path (str): root directory for KV cache "
            "files (required)\n"
            "- cuda_gpu_id (int): CUDA GPU ID for phxfs DMA "
            "(optional, default 0)\n"
            "- buffer_size_mb (int): GPU buffer size in MiB "
            "(optional, default 2048)\n"
            "- use_direct_io (bool): use O_DIRECT for I/O "
            "(optional, default true)\n"
            "- max_capacity_bytes (int): max bytes to store, "
            "0=unlimited (optional, default 0)"
        )


# ── adapter ──


class PhxL2Adapter(L2AdapterInterface):
    """Phoenix KV cache L2 adapter (asymmetric GDS).

    Store:  CPU MemoryObj → POSIX write → Phoenix storage
    Load:   Phoenix storage → phxfs_read DMA → GPU MemoryObj (or POSIX fallback)
    Lookup: hot_cache + os.path.exists
    """

    def __init__(self, config: PhxL2AdapterConfig) -> None:
        super().__init__(max_capacity_bytes=config.max_capacity_bytes)

        self._config = config
        self._base_path = config.base_path
        os.makedirs(self._base_path, exist_ok=True)

        # ── Phoenix GPU DMA setup (lazy: fallback if unavailable) ──
        self._phx_cache = None
        self._phx_base_pointer: Optional[int] = None
        self._phx_buffer_size: int = config.buffer_size_mb * 1024 * 1024
        self._phx_allocator = None

        try:
            import phxcache  # type: ignore[import-untyped]
            import torch

            self._phx_cache = phxcache.PhxCache(
                cuda_gpu_id=config.cuda_gpu_id
            )
            logger.info(
                "PhxL2Adapter: PhxCache initialized (cuda_gpu_id=%d)",
                config.cuda_gpu_id,
            )

            from lmcache.v1.memory_allocators.phx_file_memory_allocator import (
                PhxFileMemoryAllocator,
            )

            device = f"cuda:{config.cuda_gpu_id}"
            self._phx_allocator = PhxFileMemoryAllocator(
                self._phx_buffer_size,
                device=device,
                phx_cache=self._phx_cache,
            )
            self._phx_base_pointer = self._phx_allocator.base_pointer
            logger.info(
                "PhxL2Adapter: GPU buffer ready (size=%d MiB, base=0x%x)",
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

        # ── hot cache ──
        self._hot_cache: dict[str, str] = {}
        self._hot_lock = threading.Lock()

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
        """Store: CPU MemoryObj → POSIX write."""
        task_id, keys, objects = task
        import torch

        success_keys: list[ObjectKey] = []
        success_sizes: list[int] = []

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
                    # bfloat16 → view as uint8 for numpy
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
        """Load: phxfs_read DMA (GPU) or POSIX read (CPU fallback)."""
        task_id, keys, objects = task
        bitmap = Bitmap(len(keys))

        for i, (key, obj) in enumerate(zip(keys, objects)):
            try:
                with self._hot_lock:
                    fname = _object_key_to_filename(key)
                    path = self._hot_cache.get(fname)
                if path is None:
                    path = _key_to_path(self._base_path, key)
                    if not os.path.exists(path):
                        continue

                tensor = obj.raw_tensor
                if tensor is None:
                    continue
                size = tensor.nbytes

                if tensor.is_cuda and self._phx_cache is not None:
                    self._load_dma(path, tensor, size)
                else:
                    self._load_posix(path, tensor, size)

                bitmap.set(i)
                self._notify_keys_accessed([key])
            except Exception as e:
                logger.error("PhxL2Adapter load failed for %s: %s", key, e)
        with self._load_results_lock:
            self._load_results[task_id] = bitmap
        self._load_efd.notify()

    def _load_dma(self, path: str, tensor: Any, size: int) -> None:
        """Load via phxfs_read DMA to GPU buffer."""
        import phxcache

        gpu_ptr = tensor.data_ptr()
        buf_offset = gpu_ptr - self._phx_base_pointer

        flags = os.O_RDONLY
        if self._config.use_direct_io:
            flags |= os.O_DIRECT

        with phxcache.PhxFile(self._phx_cache, path, flags) as f:
            ret = f.read(
                buf=self._phx_base_pointer,
                buf_offset=buf_offset,
                nbyte=size,
                f_offset=0,
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
        with open(path, "rb") as f:
            f.readinto(buf.numpy())

    # ── lifecycle ──

    def close(self) -> None:
        self._stop_flag.set()
        self._worker_thread.join(timeout=5)

        if self._phx_allocator is not None:
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
