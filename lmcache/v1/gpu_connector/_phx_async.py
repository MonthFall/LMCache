# SPDX-License-Identifier: Apache-2.0
"""ctypes wrapper around the Phoenix phxfs C API (``libphoenix.so``).

Backend for the GDS L1 tier's async dispatch shim
(:mod:`lmcache.v1.gpu_connector._gds_async`). Phoenix (phxfs) DMAs NVMe
data straight into registered GPU buffers like cuFile/hipFile/uGDS, so the
common backend surface maps naturally:

- :func:`register_buffer` wraps ``phxfs_regmem`` (lazily opens the phxfs
  device and registers the buffer 64 KiB-aligned).
- :func:`register_handle` is a passthrough: phxfs operates on plain POSIX
  fds, so the "handle" is the fd itself.
- :class:`AsyncHandle.read_async` / ``write_async`` wrap
  ``phxfs_read`` / ``phxfs_write``.

V1 execution semantics (synchronous degradation): unlike cuFile/uGDS,
phxfs has no stream-ordered submission yet -- the DMA is host-initiated
and ``phxfs_read`` / ``phxfs_write`` return only once the transfer is
complete. This wrapper keeps the stream-ordered API shape so
:mod:`lmcache.v1.gpu_connector.gds_context` works unmodified:

- ``read_async``: the completed synchronous DMA already guarantees the
  data is visible to any subsequent op enqueued on the stream, so
  returning right away is safe.
- ``write_async``: synchronizes ``raw_stream`` first so any preceding
  gather on that stream has completed before the DMA reads the buffer.

The IO submission is funneled through :func:`_do_read` / :func:`_do_write`
so the planned stream-ordered phxgds shim (flag-slot stream bridging in
libphoenix) can later replace only those bodies.

LMCache expects ``libphoenix.so`` exposing the phxfs API (``phoenix.h``);
the library is resolved through the loader (``ldconfig`` or
``LD_LIBRARY_PATH``).
"""

# Standard
from typing import Any, Optional
import bisect
import ctypes
import ctypes.util
import os
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# --- libphoenix.so lazy loading -----------------------------------------

_lib: Optional[ctypes.CDLL] = None


def _declare_signatures(lib: ctypes.CDLL) -> None:
    """Set argtypes/restype on the phxfs symbols used by this module."""
    lib.phxfs_find_dev.argtypes = [ctypes.c_int]
    lib.phxfs_find_dev.restype = ctypes.c_int

    lib.phxfs_open.argtypes = [ctypes.c_int]
    lib.phxfs_open.restype = ctypes.c_int

    lib.phxfs_close.argtypes = [ctypes.c_int]
    lib.phxfs_close.restype = ctypes.c_int

    lib.phxfs_get_page_size.argtypes = []
    lib.phxfs_get_page_size.restype = ctypes.c_uint64

    lib.phxfs_regmem.argtypes = [
        ctypes.c_int,  # int device_id (phxfs index)
        ctypes.c_void_p,  # const void *addr (device address)
        ctypes.c_size_t,  # size_t len
        ctypes.POINTER(ctypes.c_void_p),  # void **target_addr (out)
    ]
    lib.phxfs_regmem.restype = ctypes.c_int

    lib.phxfs_deregmem.argtypes = [
        ctypes.c_int,  # int device_id (phxfs index)
        ctypes.c_void_p,  # const void *addr
        ctypes.c_size_t,  # size_t len
    ]
    lib.phxfs_deregmem.restype = ctypes.c_int

    lib.phxfs_read.argtypes = [
        ctypes.c_int,  # int fd
        ctypes.c_int,  # int device_id (phxfs index; >=0 GPU, <0 CPU)
        ctypes.c_void_p,  # void *buf
        ctypes.c_int64,  # off_t buf_offset
        ctypes.c_ssize_t,  # ssize_t nbyte
        ctypes.c_int64,  # off_t f_offset
    ]
    lib.phxfs_read.restype = ctypes.c_ssize_t

    lib.phxfs_write.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_ssize_t,
        ctypes.c_int64,
    ]
    lib.phxfs_write.restype = ctypes.c_ssize_t


def _get_lib() -> ctypes.CDLL:
    """Load ``libphoenix.so`` on first use and declare the phxfs ABI."""
    global _lib
    if _lib is not None:
        return _lib
    search = ctypes.util.find_library("phoenix")
    path = search or "libphoenix.so"
    lib = ctypes.CDLL(path)
    _declare_signatures(lib)
    _lib = lib
    return _lib


# --- Device + buffer registration state ----------------------------------
#
# phxfs devices are opened once per CUDA/HIP ordinal (find_dev + open) and
# cached; registered buffers are tracked in a sorted table so that IO can
# resolve the owning phxfs device from the buffer base pointer and so that
# deregistration passes the same (aligned) length that was registered.

_state_lock = threading.Lock()
_devices: dict[int, int] = {}
"""CUDA/HIP ordinal -> opened phxfs device index."""
_reg_bases: list[int] = []
"""Sorted registered buffer base pointers (parallel to ``_reg_entries``)."""
_reg_entries: list[tuple[int, int]] = []
"""(aligned_length, phxfs_device_index) per entry in ``_reg_bases``."""

_page_size: int = 0
"""Cached device page size in bytes (64 KiB on NVIDIA); 0 = not queried."""

_MAP_MODE_NAMES = {0: "FULL", 1: "STAGING"}


def _align_up(size: int, alignment: int) -> int:
    return (size + alignment - 1) & ~(alignment - 1)


def _get_page_size_locked(lib: ctypes.CDLL) -> int:
    """Return the (cached) device page size. Caller holds ``_state_lock``."""
    global _page_size
    if _page_size == 0:
        _page_size = int(lib.phxfs_get_page_size())
        if _page_size <= 0:
            raise RuntimeError(
                f"phxfs_get_page_size returned invalid value {_page_size}"
            )
    return _page_size


def _ensure_device_open_locked(lib: ctypes.CDLL, cuda_ordinal: int) -> int:
    """Open the phxfs device for ``cuda_ordinal`` once. Caller holds the lock.

    Returns the phxfs device index used for regmem / read / write.

    Raises:
        RuntimeError: If ``phxfs_find_dev`` or ``phxfs_open`` fails.
    """
    phxfs_dev = _devices.get(cuda_ordinal)
    if phxfs_dev is not None:
        return phxfs_dev
    phxfs_dev = int(lib.phxfs_find_dev(cuda_ordinal))
    if phxfs_dev < 0:
        raise RuntimeError(f"phxfs_find_dev({cuda_ordinal}) failed with {phxfs_dev}")
    rc = int(lib.phxfs_open(phxfs_dev))
    if rc < 0:
        raise RuntimeError(f"phxfs_open({phxfs_dev}) failed with {rc}")
    _devices[cuda_ordinal] = phxfs_dev
    try:
        map_mode = int(lib.phxfs_get_map_mode(phxfs_dev))
        mode_name = _MAP_MODE_NAMES.get(map_mode, f"unknown({map_mode})")
    except AttributeError:
        mode_name = "unknown"
    logger.info(
        "_phx_async: phxfs device %d opened for GPU %d (map_mode=%s, page_size=%d KiB)",
        phxfs_dev,
        cuda_ordinal,
        mode_name,
        _get_page_size_locked(lib) // 1024,
    )
    return phxfs_dev


def _lookup_device(buf_base: int) -> int:
    """Resolve the phxfs device whose registration covers ``buf_base``.

    Returns:
        The phxfs device index for the registration containing ``buf_base``.

    Raises:
        RuntimeError: If ``buf_base`` does not fall inside any registered
            buffer.
    """
    with _state_lock:
        idx = bisect.bisect_right(_reg_bases, buf_base) - 1
        if idx < 0:
            raise RuntimeError(
                f"buf_base 0x{buf_base:x} is not inside any registered phxfs buffer"
            )
        base = _reg_bases[idx]
        aligned_len, phxfs_dev = _reg_entries[idx]
        if buf_base >= base + aligned_len:
            raise RuntimeError(
                f"buf_base 0x{buf_base:x} is not inside any registered phxfs buffer"
            )
    return phxfs_dev


# --- Backend surface (contract of _gds_async) ----------------------------


def register_handle(fd: int) -> int:
    """Accept an open fd for phxfs IO and return the "handle".

    phxfs reads and writes plain POSIX fds (no library-side handle
    registration), so the fd itself is the handle; :class:`AsyncHandle`
    round-trips it and closes the fd on ``close()``. Loads ``libphoenix``
    eagerly so a missing library fails at slab setup, not at first DMA.

    Args:
        fd: Open slab-file descriptor (as created by
            :meth:`gds_context.GDSContext.initialize`).

    Returns:
        ``fd`` unchanged.
    """
    _get_lib()
    return fd


def deregister_handle(handle: int) -> None:
    """Reverse of :func:`register_handle`: nothing to do for phxfs."""
    return


def register_buffer(buf: torch.Tensor) -> None:
    """Register a device tensor with phxfs for GDS DMA.

    Opens the phxfs device for the tensor's GPU on first use, then
    registers the buffer with ``phxfs_regmem``. phxfs requires the
    registration length to be device-page aligned (64 KiB on NVIDIA), so
    the length is rounded up; the IO size is independent of the
    registration length and never exceeds the buffer.

    Args:
        buf: Contiguous GPU tensor (a <=16 MiB slice of a staging buffer,
            as passed by :meth:`gds_context.GDSContext.register_gpu_buffer`).

    Raises:
        ValueError: If ``buf`` is not a GPU tensor or is empty.
        RuntimeError: If the phxfs device cannot be opened or the
            registration is rejected.
    """
    if not buf.is_cuda:
        raise ValueError("register_buffer: tensor must be on a CUDA or ROCm GPU")
    nbytes = buf.numel() * buf.element_size()
    if nbytes == 0:
        raise ValueError("register_buffer: tensor is empty")
    base = buf.data_ptr()
    cuda_ordinal = buf.device.index
    if cuda_ordinal is None:
        cuda_ordinal = torch.cuda.current_device()
    lib = _get_lib()
    with _state_lock:
        phxfs_dev = _ensure_device_open_locked(lib, cuda_ordinal)
        page_size = _get_page_size_locked(lib)
        aligned_len = _align_up(nbytes, page_size)
        # Reject a mismatched duplicate before touching phxfs (an exact
        # duplicate is reference-counted by phxfs and allowed through).
        idx = bisect.bisect_left(_reg_bases, base)
        if idx < len(_reg_bases) and _reg_bases[idx] == base:
            prev_len, _ = _reg_entries[idx]
            if prev_len != aligned_len:
                raise RuntimeError(
                    f"register_buffer: 0x{base:x} already registered with "
                    f"a different length ({prev_len} != {aligned_len})"
                )
        target_addr = ctypes.c_void_p()
        rc = int(
            lib.phxfs_regmem(phxfs_dev, base, aligned_len, ctypes.byref(target_addr))
        )
        if rc < 0:
            raise RuntimeError(
                f"phxfs_regmem failed with {rc}: device={phxfs_dev}, "
                f"addr=0x{base:x}, len={aligned_len}"
            )
        # NOTE: target_addr is an internal host-mapped handle; IO must use
        # the original device address, so it is deliberately not recorded.
        if idx < len(_reg_bases) and _reg_bases[idx] == base:
            # Exact duplicate: phxfs reference-counts it; keep one entry.
            return
        _reg_bases.insert(idx, base)
        _reg_entries.insert(idx, (aligned_len, phxfs_dev))
    logger.debug(
        "_phx_async: registered 0x%x..0x%x on phxfs device %d",
        base,
        base + aligned_len,
        phxfs_dev,
    )


def deregister_buffer(buf: torch.Tensor) -> None:
    """Reverse of :func:`register_buffer`.

    Deregisters the buffer's registration with ``phxfs_deregmem`` using
    the same aligned length that was registered. Unregistered buffers are
    ignored (matching the tolerance of the cuFile path teardown).

    Args:
        buf: The tensor previously passed to :func:`register_buffer`.

    Raises:
        RuntimeError: If ``phxfs_deregmem`` fails.
    """
    lib = _get_lib()
    base = buf.data_ptr()
    with _state_lock:
        idx = bisect.bisect_left(_reg_bases, base)
        if idx >= len(_reg_bases) or _reg_bases[idx] != base:
            return
        aligned_len, phxfs_dev = _reg_entries[idx]
        rc = int(lib.phxfs_deregmem(phxfs_dev, base, aligned_len))
        if rc < 0:
            raise RuntimeError(
                f"phxfs_deregmem failed with {rc}: device={phxfs_dev}, "
                f"addr=0x{base:x}, len={aligned_len}"
            )
        del _reg_bases[idx]
        del _reg_entries[idx]


def register_stream(raw_stream: int) -> None:
    """No-op in V1: the synchronous execution path needs no stream state.

    V2 (stream-ordered phxgds shim) will allocate a flag slot per stream
    here.

    Args:
        raw_stream: Raw CUDA/ROCm stream handle (unused in V1).
    """
    del raw_stream


def deregister_stream(raw_stream: int) -> None:
    """No-op in V1 (see :func:`register_stream`)."""
    del raw_stream


def close_driver() -> None:
    """Release every phxfs registration and close all opened devices.

    Sweeps any registration still left in the table (the normal teardown
    path deregisters each buffer via :func:`deregister_buffer`), then
    closes every opened phxfs device.
    """
    lib = _get_lib()
    with _state_lock:
        for base, (aligned_len, phxfs_dev) in zip(
            _reg_bases, _reg_entries, strict=True
        ):
            try:
                rc = int(lib.phxfs_deregmem(phxfs_dev, base, aligned_len))
                if rc < 0:
                    logger.warning(
                        "close_driver: phxfs_deregmem(0x%x) failed with %d",
                        base,
                        rc,
                    )
            except Exception as e:  # noqa: BLE001 - teardown sweep
                logger.warning("close_driver: phxfs_deregmem raised %s", e)
        _reg_bases.clear()
        _reg_entries.clear()
        for phxfs_dev in sorted(set(_devices.values())):
            rc = int(lib.phxfs_close(phxfs_dev))
            if rc < 0:
                logger.warning(
                    "close_driver: phxfs_close(%d) failed with %d",
                    phxfs_dev,
                    rc,
                )
        _devices.clear()


# --- IO submission (V2 switchover point) ---------------------------------


def _do_read(
    fd: int, buf_base: int, size: int, file_offset: int, buf_offset: int
) -> int:
    """Perform one read DMA; return when the data is in the GPU buffer.

    V1: synchronous ``phxfs_read`` -- returning from this function means
    the transfer is complete, which satisfies the async contract (data is
    visible to anything the caller enqueues on the stream afterwards).
    V2: replace the body with the stream-ordered phxgds submission.
    """
    phxfs_dev = _lookup_device(buf_base)
    result = int(
        _get_lib().phxfs_read(fd, phxfs_dev, buf_base, buf_offset, size, file_offset)
    )
    if result < 0:
        raise RuntimeError(
            f"phxfs_read failed with {result}: fd={fd}, device={phxfs_dev}, "
            f"buf_offset={buf_offset}, nbyte={size}, f_offset={file_offset}"
        )
    return result


def _do_write(
    fd: int,
    buf_base: int,
    size: int,
    file_offset: int,
    buf_offset: int,
    raw_stream: int,
) -> int:
    """Perform one write DMA after the stream's prior work completes.

    V1: synchronize ``raw_stream`` first so any gather enqueued before
    this call has finished writing the GPU buffer, then issue the
    synchronous ``phxfs_write``. V2: replace the body with the
    stream-ordered phxgds submission (fence + DMA).
    """
    torch.cuda.ExternalStream(raw_stream).synchronize()
    phxfs_dev = _lookup_device(buf_base)
    result = int(
        _get_lib().phxfs_write(fd, phxfs_dev, buf_base, buf_offset, size, file_offset)
    )
    if result < 0:
        raise RuntimeError(
            f"phxfs_write failed with {result}: fd={fd}, device={phxfs_dev}, "
            f"buf_offset={buf_offset}, nbyte={size}, f_offset={file_offset}"
        )
    return result


# --- Submission + AsyncHandle --------------------------------------------


class Submission:
    """One completed (V1) or in-flight (V2) phxfs read / write.

    Mirrors :class:`_cufile_async.Submission`: holds the transfer
    parameters and the ``bytes_done`` result storage. V2 will hand the
    same ctypes storage to the phxgds shim, so the keep-alive-until-sync
    contract documented there applies unchanged.
    """

    __slots__ = ("_size", "_file_offset", "_buf_offset", "_bytes_done")

    def __init__(self, size: int, file_offset: int, buf_offset: int) -> None:
        self._size = ctypes.c_size_t(size)
        self._file_offset = ctypes.c_int64(file_offset)
        self._buf_offset = ctypes.c_int64(buf_offset)
        self._bytes_done = ctypes.c_int64(0)

    @property
    def bytes_done(self) -> int:
        return self._bytes_done.value


class AsyncHandle:
    """Slab-file handle wrapper, API-compatible with _cufile_async.AsyncHandle."""

    __slots__ = ("_fd", "_handle", "path", "writable")

    def __init__(
        self,
        device_path: str,
        writable: bool = True,
    ) -> None:
        fd = os.open(device_path, os.O_RDWR)
        try:
            handle = register_handle(fd)
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        self._handle = handle
        self.path = device_path
        self.writable = writable

    @classmethod
    def from_fd(
        cls,
        fd: int,
        handle: int,
        path: str,
        writable: bool = False,
    ) -> "AsyncHandle":
        """Wrap an already-opened fd (``handle`` is the fd itself for phxfs)."""
        obj = cls.__new__(cls)
        obj._fd = fd
        obj._handle = handle
        obj.path = path
        obj.writable = writable
        return obj

    @property
    def fd(self) -> int:
        return self._fd

    def read_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Read ``size`` bytes from the slab into the registered GPU buffer.

        V1 executes the DMA synchronously; the returned :class:`Submission`
        already carries the final ``bytes_done``.

        Args:
            buf_base: Base pointer of a registered GPU buffer region.
            size: Transfer length in bytes.
            file_offset: Slab-file offset to read from.
            buf_offset: Offset within the GPU buffer region.
            raw_stream: Raw stream handle (unused by the V1 read path).

        Returns:
            The (completed) submission.

        Raises:
            RuntimeError: If ``buf_base`` is not registered or the DMA fails.
        """
        del raw_stream  # V1 read needs no stream ordering (data-ready on return)
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        sub._bytes_done.value = _do_read(
            fd=self._fd,
            buf_base=buf_base,
            size=size,
            file_offset=file_offset,
            buf_offset=buf_offset,
        )
        return sub

    def write_async(
        self,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        raw_stream: int,
    ) -> Submission:
        """Write ``size`` bytes from the registered GPU buffer to the slab.

        V1 synchronizes ``raw_stream`` (draining any prior gather) before
        issuing the synchronous DMA.

        Args:
            buf_base: Base pointer of a registered GPU buffer region.
            size: Transfer length in bytes.
            file_offset: Slab-file offset to write to.
            buf_offset: Offset within the GPU buffer region.
            raw_stream: Raw stream handle whose prior work must complete
                before the DMA reads the buffer.

        Returns:
            The (completed) submission.

        Raises:
            RuntimeError: If ``buf_base`` is not registered or the DMA fails.
        """
        sub = Submission(size=size, file_offset=file_offset, buf_offset=buf_offset)
        sub._bytes_done.value = _do_write(
            fd=self._fd,
            buf_base=buf_base,
            size=size,
            file_offset=file_offset,
            buf_offset=buf_offset,
            raw_stream=raw_stream,
        )
        return sub

    def close(self) -> None:
        if self._fd < 0:
            return
        try:
            deregister_handle(self._handle)
        finally:
            try:
                os.close(self._fd)
            finally:
                self._fd = -1

    def __enter__(self) -> "AsyncHandle":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
