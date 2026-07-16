# SPDX-License-Identifier: Apache-2.0

"""Phoenix storage backend for LMCache.

Implements an **asymmetric** I/O model:

* **Store**: CPU MemoryObj → POSIX write → Phoenix storage.  Data already
  resides in CPU memory (via D2H from vLLM), so we write it directly with
  POSIX I/O — no GPU buffer needed.  ``get_allocator_backend()`` returns
  ``local_cpu_backend`` so that ``batched_put`` reuses the CPU MemoryObj
  and skips ``allocate_and_copy_objects``.

* **Retrieve**: Phoenix storage → ``phxfs_read_async`` DMA → GPU MemoryObj.
  A dedicated CUDA stream batches all async reads, then synchronizes once.
  This is the GDS value proposition — NVMe-to-GPU DMA bypassing host memory.

The file format is identical to ``GdsBackend`` (safetensors-style metadata
header + raw KV data), so the metadata management code is reused from
``gds_backend.py``.
"""

# Standard
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import asyncio
import json
import os
import struct
import threading
import time
import urllib.parse

# Third Party
import aiofile
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import (
    CacheEngineKey,
    DiskCacheMetadata,
    _lmcache_nvtx_annotate,
    parse_cache_key,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.gds_backend import (
    _DATA_FILE_SUFFIX,
    _METADATA_FILE_SUFFIX,
    _METADATA_MAX_SIZE,
    _METADATA_VERSION,
    UnsupportedMetadataVersion,
    pack_metadata,
    unpack_metadata,
    save_metadata,
    rand_suffix,
)
from lmcache.v1.storage_backend.path_sharder import PathSharder

logger = init_logger(__name__)

_DEFAULT_THREAD_COUNT = 4


class PhxBackend(AllocatorBackendInterface):
    """Phoenix storage backend using asymmetric GDS (POSIX store + DMA retrieve).

    Configuration (in ``LMCacheEngineConfig``):

    * ``phx_path``: Phoenix storage path (comma-separated for multi-path).
    * ``phx_path_sharding``: Sharding strategy (``"by_gpu"`` etc.).
    * ``phx_buffer_size``: GPU buffer size in MB (retrieve only).

    Requires ``LocalCPUBackend`` for the asymmetric store path.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = "cuda",
        local_cpu_backend=None,
    ):
        assert dst_device.startswith("cuda"), (
            f"PhxBackend requires a CUDA device, got '{dst_device}'"
        )
        super().__init__(dst_device=dst_device)

        self.config = config
        self.loop = loop
        self.dst_device = dst_device
        self.local_cpu_backend = local_cpu_backend

        assert config.phx_path is not None, "Need to specify phx_path for PhxBackend"
        assert local_cpu_backend is not None, (
            "PhxBackend (asymmetric mode) requires local_cpu_backend"
        )

        # Multi-path sharding (same pattern as GdsBackend)
        sharder = PathSharder(
            raw_csv=config.phx_path,
            strategy=config.phx_path_sharding,
            dst_device=dst_device,
        )
        self.phx_paths = sharder.all_paths
        self.phx_path = sharder.selected

        logger.info(
            "PhxBackend using path '%s' (%d path(s) configured)",
            self.phx_path,
            len(self.phx_paths),
        )

        # Initialize PhxCache (Phoenix device connection)
        import phxcache

        self._phxcache_module = phxcache
        cuda_gpu_id = torch.device(dst_device).index
        if cuda_gpu_id is None:
            cuda_gpu_id = 0
        self.phx_cache = phxcache.PhxCache(cuda_gpu_id=cuda_gpu_id)

        # Initialize GPU memory allocator (for retrieve path only)
        self.memory_allocator = self.initialize_allocator(config, metadata)

        # Slab / base pointer support
        if hasattr(self.memory_allocator, "target_base"):
            self.phx_target_base = self.memory_allocator.target_base
            self.phx_base_pointer = self.memory_allocator.base_pointer
        else:
            self.phx_target_base = None
            self.phx_base_pointer = None

        # File format (compatible with GdsBackend)
        self.data_suffix = _DATA_FILE_SUFFIX

        # Metadata cache: key -> DiskCacheMetadata
        self.hot_lock = threading.Lock()
        self.hot_cache: OrderedDict[CacheEngineKey, DiskCacheMetadata] = OrderedDict()
        self.metadata_dirs: set[str] = set()

        # Put task tracking
        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        # Retrieve: dedicated CUDA stream for batched phxfs_read_async
        self.load_stream = torch.cuda.Stream(device=dst_device)

        # Store: thread pool for async POSIX writes
        self._thread_pool = ThreadPoolExecutor(
            max_workers=_DEFAULT_THREAD_COUNT,
            thread_name_prefix="phx-io",
        )

        # Retry settings (same as GdsBackend)
        self.max_alloc_attempts = (config.extra_config or {}).get(
            "max_alloc_attempts", 10
        )
        self.alloc_attempt_delay_secs = (config.extra_config or {}).get(
            "allocation_attempt_delay_secs", 0.1
        )

        # Create storage directories
        for p in self.phx_paths:
            os.makedirs(p, exist_ok=True)

        # Scan existing metadata on disk (async, non-blocking init)
        self.save_metadata_tasks: set[asyncio.Task] = set()
        self._scan_metadata_future = asyncio.run_coroutine_threadsafe(
            self._scan_metadata(), self.loop
        )

        self._debug_asserts = False
        self._use_noatime = True

        self.stats = None

    # ------------------------------------------------------------------
    # Metadata scanning (reused from GdsBackend pattern)
    # ------------------------------------------------------------------

    async def _scan_metadata(self):
        tasks = []
        start = time.perf_counter()
        for p in self.phx_paths:
            with os.scandir(p) as it:
                for entry in it:
                    if not entry.is_dir():
                        continue
                    l1_dir = os.path.basename(entry.name)
                    if len(l1_dir) != 2:
                        continue
                    tasks.append(
                        asyncio.to_thread(
                            self._scan_metadata_subdir,
                            os.path.join(p, l1_dir),
                            l1_dir,
                        )
                    )
        await asyncio.gather(*tasks)
        end = time.perf_counter()
        logger.info(
            "PhxBackend: Read %d cache entries from persistent storage "
            "in %.2f seconds",
            len(self.hot_cache),
            end - start,
        )

    def _scan_metadata_subdir(self, path, l1_dir):
        target_suffix = self.data_suffix + _METADATA_FILE_SUFFIX
        with os.scandir(path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                l2_dir = os.path.basename(entry.name)
                if len(l2_dir) != 2:
                    continue
                with os.scandir(os.path.join(path, l2_dir)) as it2:
                    for fentry in it2:
                        if not fentry.is_file():
                            continue
                        if not fentry.name.endswith(target_suffix):
                            continue
                        filename = os.path.basename(fentry.name)
                        key_str = urllib.parse.unquote(
                            filename[: -len(target_suffix)]
                        )
                        try:
                            key = parse_cache_key(key_str)
                        except ValueError as e:
                            logger.error(
                                "Filename %s can't be converted back into "
                                "cache key: %s",
                                filename,
                                e,
                            )
                            continue
                        try:
                            self._read_metadata(key, fentry.path, l1_dir + l2_dir)
                        except UnsupportedMetadataVersion:
                            logger.error(
                                "Unsupported metadata version for %s; ignoring",
                                fentry.path,
                            )
                        except Exception:
                            logger.error(
                                "Failed to read metadata file %s; raising",
                                fentry.path,
                                exc_info=True,
                            )
                            raise

    def _read_metadata_info(self, filename: str):
        if self._use_noatime:
            try:
                fd = os.open(filename, os.O_RDONLY | os.O_NOATIME)
            except (PermissionError, AttributeError, OSError):
                self._use_noatime = False
                fd = os.open(filename, os.O_RDONLY)
        else:
            fd = os.open(filename, os.O_RDONLY)
        try:
            buf = os.read(fd, _METADATA_MAX_SIZE)
        finally:
            os.close(fd)
        return unpack_metadata(buf)

    def _read_metadata(self, key: CacheEngineKey, filename: str, subdir_key: str):
        shape, dtype, size, fmt, extra_metadata = self._read_metadata_info(filename)
        if extra_metadata["lmcache_version"] != str(_METADATA_VERSION):
            raise UnsupportedMetadataVersion("unhandled lmcache metadata")
        metadata = DiskCacheMetadata(
            filename.removesuffix(_METADATA_FILE_SUFFIX),
            size,
            shape,
            dtype,
            None,
            fmt,
        )
        with self.hot_lock:
            self.metadata_dirs.add(subdir_key)
            self.hot_cache[key] = metadata
        return metadata

    # ------------------------------------------------------------------
    # Path management
    # ------------------------------------------------------------------

    def _key_to_path(
        self,
        key: CacheEngineKey,
        base_path: Optional[str] = None,
    ) -> Tuple[str, str, str, str]:
        hash_str = str(key.chunk_hash)
        l1_dir = hash_str[:2]
        l2_dir = hash_str[2:4]
        key_str = key.to_string()
        if base_path is None:
            base_path = self.phx_path
        return (
            os.path.join(
                base_path,
                l1_dir,
                l2_dir,
                urllib.parse.quote(key_str, safe="") + self.data_suffix,
            ),
            l1_dir + l2_dir,
            l1_dir,
            l2_dir,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.hot_lock:
            res = key in self.hot_cache
        if res:
            return True
        if self._try_to_read_metadata(key):
            return True
        return False

    def _try_to_read_metadata(self, key: CacheEngineKey) -> Optional[DiskCacheMetadata]:
        for p in self.phx_paths:
            path, subdir_key, _, _ = self._key_to_path(key, base_path=p)
            path += _METADATA_FILE_SUFFIX
            if os.path.exists(path):
                try:
                    return self._read_metadata(key, path, subdir_key)
                except FileNotFoundError:
                    logger.warning(
                        "File not found for key %s at %s",
                        key.to_string(),
                        path,
                    )
                except PermissionError:
                    logger.warning("Permission denied for %s", path)
                except UnsupportedMetadataVersion:
                    logger.error("Unsupported metadata version for %s", path)
                except (OSError, IOError) as e:
                    logger.error("Failed to read metadata %s: %s", path, e)

        return None

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        hit_chunks = 0
        for key in keys:
            if not self.contains(key, pin):
                break
            hit_chunks += 1
        return hit_chunks

    # ------------------------------------------------------------------
    # Asymmetric mode: allocator backend
    # ------------------------------------------------------------------

    def get_allocator_backend(self):
        """Return local_cpu_backend so batched_put reuses CPU MemoryObjs.

        This is the key to the asymmetric design: store path gets CPU
        MemoryObjs (already filled by D2H from vLLM) and writes them via
        POSIX — no GPU buffer or H2D copy needed.
        """
        if self.local_cpu_backend is not None:
            return self.local_cpu_backend
        return self

    def get_memory_allocator(self):
        return self.memory_allocator

    # ------------------------------------------------------------------
    # Store path: POSIX write from CPU MemoryObj
    # ------------------------------------------------------------------

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Future:
        assert memory_obj.tensor is not None
        memory_obj.ref_count_up()

        with self.put_lock:
            self.put_tasks.add(key)

        future = asyncio.run_coroutine_threadsafe(
            self._async_save_bytes_to_disk(key, memory_obj, on_complete_callback),
            self.loop,
        )
        return future

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Union[List[Future], None]:
        futures = []
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            future = self.submit_put_task(
                key, memory_obj, on_complete_callback=on_complete_callback
            )
            futures.append(future)
        return futures

    async def _async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        try:
            kv_chunk = memory_obj.tensor
            assert kv_chunk is not None
            path, subdir_key, l1_dir, l2_dir = self._key_to_path(key)

            if subdir_key not in self.metadata_dirs:
                os.makedirs(os.path.join(self.phx_path, l1_dir, l2_dir), exist_ok=True)
                self.metadata_dirs.add(subdir_key)

            tmp = ".tmp" + rand_suffix(8)
            fmt = memory_obj.metadata.fmt

            try:
                metadata = await asyncio.to_thread(
                    self._save_phx, path, tmp, kv_chunk, fmt
                )
            except Exception as e:
                logger.error(
                    "PhxBackend write failed for key %s at path %s: %s",
                    key.to_string(),
                    path,
                    e,
                    exc_info=True,
                )
                return

            self.insert_key(key, memory_obj)

            try:
                task = asyncio.create_task(
                    save_metadata(path + _METADATA_FILE_SUFFIX, tmp, metadata)
                )
                self.save_metadata_tasks.add(task)
                task.add_done_callback(self.save_metadata_tasks.discard)
                task.add_done_callback(
                    lambda t: self._handle_metadata_write_completion(t, key, path)
                )
            except Exception as e:
                logger.error(
                    "Metadata write failed for key %s at %s: %s",
                    key.to_string(),
                    path + _METADATA_FILE_SUFFIX,
                    e,
                    exc_info=True,
                )
                with self.hot_lock:
                    self.hot_cache.pop(key, None)
                return
        finally:
            memory_obj.ref_count_down()
            with self.put_lock:
                self.put_tasks.discard(key)

        if on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.error(
                    "on_complete_callback failed for key %s: %s",
                    key.to_string(),
                    e,
                    exc_info=True,
                )

    def _handle_metadata_write_completion(
        self, task: asyncio.Task, key: CacheEngineKey, path: str
    ) -> None:
        try:
            exception = task.exception()
            if exception is not None:
                logger.error(
                    "Metadata write task failed for key %s at %s: %s",
                    key.to_string(),
                    path + _METADATA_FILE_SUFFIX,
                    exception,
                    exc_info=exception,
                )
                with self.hot_lock:
                    self.hot_cache.pop(key, None)
        except Exception as e:
            logger.error(
                "Error checking metadata write task for key %s: %s",
                key.to_string(),
                e,
                exc_info=True,
            )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _save_phx(self, path: str, tmp: str, kv_chunk: torch.Tensor, fmt: MemoryFormat):
        """POSIX write from CPU MemoryObj (asymmetric store path).

        File layout (identical to GdsBackend):
        ``[0, _METADATA_MAX_SIZE)``  — metadata (POSIX write)
        ``[_METADATA_MAX_SIZE, end)`` — KV data (POSIX write)
        """
        tmp_path = path + tmp
        offset = _METADATA_MAX_SIZE
        metadata = pack_metadata(kv_chunk, fmt=fmt, lmcache_version=str(_METADATA_VERSION))

        try:
            with open(tmp_path, "wb") as f:
                f.write(metadata)
                f.seek(offset)
                # numpy does not support bfloat16; view as uint8 for a
                # zero-copy raw-byte write (same underlying storage).
                f.write(kv_chunk.contiguous().view(torch.uint8).numpy())
        except Exception as e:
            logger.error("Error saving %s: %s", tmp_path, e, exc_info=True)
            raise e

        os.rename(tmp_path, path)
        return metadata

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path, _, _, _ = self._key_to_path(key)
        size = memory_obj.get_physical_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        with self.hot_lock:
            self.hot_cache[key] = DiskCacheMetadata(
                path, size, shape, dtype, None, fmt
            )

    # ------------------------------------------------------------------
    # Retrieve path: phxfs_read_async DMA to GPU MemoryObj
    # ------------------------------------------------------------------

    def _get_dma_addr(self, memory_obj: MemoryObj) -> int:
        """Return the raw GPU pointer for a MemoryObj's buffer.

        ``phxfs_regmem`` registers the *original* GPU virtual address; all
        subsequent ``phxfs_read`` / ``phxfs_read_async`` calls must pass the
        same GPU virtual address (possibly with an offset *within* the
        registered block) so that phxfs can look up the P2P mapping.

        The ``target_base`` returned by ``regmem()`` is a DMA-side internal
        address used by the phxfs driver — it is NOT the value to pass to
        I/O calls.
        """
        tensor = memory_obj.tensor
        assert tensor is not None
        return tensor.data_ptr()

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.hot_lock:
            entry = self.hot_cache.get(key)
        if entry is None:
            return None

        path = entry.path
        dtype = entry.dtype
        shape = entry.shape
        fmt = entry.fmt
        assert dtype is not None
        assert shape is not None
        assert fmt is not None

        memory_obj = self.memory_allocator.allocate(shape, dtype, fmt=fmt)
        if memory_obj is None:
            logger.error("Memory allocation failed during sync disk load.")
            return None

        return self._load_bytes_from_disk_with_memory(key, path, memory_obj)

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        # TODO: implement prefetch via phxfs_read_async on background stream
        return None

    def _load_bytes_from_disk_with_memory(
        self,
        key: CacheEngineKey,
        path: str,
        memory_obj: Optional[MemoryObj],
    ) -> Optional[MemoryObj]:
        if memory_obj is None or not memory_obj.is_valid():
            return None

        file_offset = _METADATA_MAX_SIZE
        gpu_ptr = self._get_dma_addr(memory_obj)
        size = memory_obj.get_size()

        if self.phx_base_pointer is not None:
            buf_offset = gpu_ptr - self.phx_base_pointer
            buf = self.phx_base_pointer
        else:
            buf_offset = 0
            buf = gpu_ptr

        try:
            with self._phxcache_module.PhxFile(
                self.phx_cache, path, os.O_RDONLY | os.O_DIRECT
            ) as f:
                ret = f.read(buf, buf_offset, size, file_offset)
        except Exception as e:
            logger.error("PhxBackend read failed for %s: %s", path, e, exc_info=True)
            memory_obj.ref_count_down()
            return None

        if ret != size:
            if ret < 0:
                logger.error("Error loading %s: ret=%d, removing entry", path, ret)
                with self.hot_lock:
                    self.hot_cache.pop(key)
            else:
                logger.error(
                    "Error loading %s: got %d bytes out of %d",
                    path,
                    ret,
                    size,
                )
            memory_obj.ref_count_down()
            return None

        return memory_obj

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        """Batched retrieve: allocate GPU buffers, DMA read from Phoenix storage.

        Uses sync ``phxfs_read`` (not ``phxfs_read_async``) because the async
        variant does not support writing to an arbitrary offset within a
        pre-registered memory region — it always targets the base address.
        The sync API's ``buf_offset`` parameter correctly handles this.
        """
        # Step 1: Look up metadata for all keys
        paths: list[str | None] = []
        dtypes: list[torch.dtype | None] = []
        shapes: list[torch.Size | None] = []
        fmts: list[MemoryFormat | None] = []

        with self.hot_lock:
            for key in keys:
                entry = self.hot_cache.get(key)
                if entry is None:
                    logger.error("Lookup failed during get_blocking for %s", key)
                    paths.append(None)
                    dtypes.append(None)
                    shapes.append(None)
                    fmts.append(None)
                    continue
                paths.append(entry.path)
                dtypes.append(entry.dtype)
                shapes.append(entry.shape)
                fmts.append(entry.fmt)

        # Step 2: Allocate GPU MemoryObjs for all valid keys
        memory_objs: list[MemoryObj | None] = []
        for dtype, shape, path, fmt in zip(dtypes, shapes, paths, fmts, strict=True):
            if path is None:
                memory_objs.append(None)
                continue
            mem_obj = self.memory_allocator.allocate(shape, dtype, fmt=fmt)
            if mem_obj is None:
                logger.error("Memory allocation failed for %s", path)
            memory_objs.append(mem_obj)

        # Step 3: Sequential sync reads via phxfs_read
        with torch.cuda.stream(self.load_stream):
            for i, mem_obj in enumerate(memory_objs):
                if mem_obj is None:
                    continue
                path = paths[i]
                assert path is not None

                gpu_ptr = self._get_dma_addr(mem_obj)
                size = mem_obj.get_size()
                file_offset = _METADATA_MAX_SIZE

                # phxfs_read expects the base of the registered region and
                # a buf_offset within it.
                if self.phx_base_pointer is not None:
                    buf = self.phx_base_pointer
                    buf_offset = gpu_ptr - self.phx_base_pointer
                else:
                    buf = gpu_ptr
                    buf_offset = 0

                try:
                    with self._phxcache_module.PhxFile(
                        self.phx_cache, path, os.O_RDONLY | os.O_DIRECT
                    ) as f:
                        ret = f.read(buf, buf_offset, size, file_offset)
                except Exception as e:
                    logger.error("PhxBackend read failed for %s: %s", path, e,
                                 exc_info=True)
                    mem_obj.ref_count_down()
                    memory_objs[i] = None
                    continue

                if ret != size:
                    if ret < 0:
                        logger.error("Error loading %s: ret=%d, removing entry",
                                     path, ret)
                        with self.hot_lock:
                            self.hot_cache.pop(keys[i])
                    else:
                        logger.error("Error loading %s: got %d bytes out of %d",
                                     path, ret, size)
                    mem_obj.ref_count_down()
                    memory_objs[i] = None

        return memory_objs

    # ------------------------------------------------------------------
    # Pin / Remove
    # ------------------------------------------------------------------

    def pin(self, key: CacheEngineKey) -> bool:
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        return False

    def remove(self, key: CacheEngineKey, force: bool = True):
        with self.hot_lock:
            entry = self.hot_cache.pop(key, None)
        if entry is not None:
            path = entry.path
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning("Failed to remove %s: %s", path, e)
            try:
                os.remove(path + _METADATA_FILE_SUFFIX)
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.warning("Failed to remove %s: %s", path + _METADATA_FILE_SUFFIX, e)
        return True

    # ------------------------------------------------------------------
    # Allocator interface
    # ------------------------------------------------------------------

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ):
        assert config.phx_buffer_size is not None, (
            "Need to specify phx_buffer_size for PhxBackend"
        )
        # First Party
        from lmcache.v1.memory_allocators.phx_file_memory_allocator import (
            PhxFileMemoryAllocator,
        )

        return PhxFileMemoryAllocator(
            config.phx_buffer_size * 1024**2,
            device=self.dst_device,
            phx_cache=self.phx_cache,
        )

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        if eviction:
            logger.warning("PhxBackend does not support eviction")

        max_attempts = self.max_alloc_attempts if busy_loop else 1
        num_attempts = 0

        while True:
            memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
            if memory_obj is not None:
                return memory_obj

            num_attempts += 1
            if num_attempts < max_attempts:
                if self.alloc_attempt_delay_secs > 0:
                    time.sleep(self.alloc_attempt_delay_secs)
            else:
                break

        logger.warning("PhxBackend allocation failed after %d attempts", num_attempts)
        return None

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        if eviction:
            logger.warning("PhxBackend does not support eviction")

        max_attempts = self.max_alloc_attempts if busy_loop else 1
        num_attempts = 0

        while True:
            memory_objs = self.memory_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt
            )
            if memory_objs is not None:
                return memory_objs

            num_attempts += 1
            if num_attempts < max_attempts:
                if self.alloc_attempt_delay_secs > 0:
                    time.sleep(self.alloc_attempt_delay_secs)
            else:
                break

        logger.warning("PhxBackend batched allocation failed after %d attempts", num_attempts)
        return None

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        # Wait for metadata scan to complete
        try:
            self._scan_metadata_future.result(timeout=30)
        except Exception as e:
            logger.warning("Exception while waiting for metadata scan: %s", e)

        # Wait for pending metadata write tasks
        if self.save_metadata_tasks:

            async def _drain_tasks() -> None:
                await asyncio.gather(
                    *self.save_metadata_tasks, return_exceptions=True
                )
                self.save_metadata_tasks.clear()

            try:
                drain = asyncio.run_coroutine_threadsafe(_drain_tasks(), self.loop)
                drain.result(timeout=30)
            except Exception as e:
                logger.warning("Exception while draining metadata tasks: %s", e)

        # Close memory allocator (deregisters GPU memory)
        self.memory_allocator.close()

        # Shut down thread pool
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)

        # Close Phoenix device connection
        try:
            self.phx_cache.close()
        except Exception as e:
            logger.warning("Exception while closing PhxCache: %s", e)

        logger.info("PhxBackend closed.")
