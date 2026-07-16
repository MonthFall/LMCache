# SPDX-License-Identifier: Apache-2.0

"""GPU memory allocator that registers memory with Phoenix phxfs for DMA.

This allocator is the Phoenix equivalent of ``CuFileMemoryAllocator``. It
pre-allocates a contiguous GPU memory buffer and registers it with
``phxfs_regmem`` so that ``phxfs_read`` / ``phxfs_read_async`` can DMA directly
between NVMe (via Phoenix) and GPU memory, bypassing the host CPU.
"""

# Standard
import ctypes

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.memory_allocators.gpu_memory_allocator import GPUMemoryAllocator


class PhxFileMemoryAllocator(GPUMemoryAllocator):
    """GPU memory allocator backed by Phoenix ``phxfs_regmem``.

    Differences from ``CuFileMemoryAllocator``:

    * Uses ``phxfs_regmem`` (via the ``phxcache`` pybind11 extension) instead
      of ``cuFileBufRegister``.
    * ``phxfs_regmem`` returns a *target_addr* (DMA-mapped address) that is
      stored for diagnostics but **not** passed to I/O calls.  All
      ``phxfs_read`` calls use the original ``tensor.data_ptr()`` (the raw
      GPU virtual address) with ``buf_offset`` for the slab offset.
    * Alignment is 64 KiB (``GPU_PAGE_SIZE``) rather than 4096.
    * The ``device`` parameter must match the physical GPU that the
      ``PhxCache`` instance maps to (e.g. GPU 4 → phxfs_dev2).

    :param size: Size of the GPU memory pool in bytes.
    :param device: Torch device string (e.g. ``"cuda:4"``).  Must match the
        GPU that ``phx_cache`` is bound to.
    :param phx_cache: A ``phxcache.PhxCache`` instance with an open Phoenix
        device connection.  Required.
    """

    def __init__(self, size: int, device=None, phx_cache=None) -> None:
        assert phx_cache is not None, (
            "PhxFileMemoryAllocator requires a phxcache.PhxCache instance"
        )

        if device is None:
            if torch_dev.is_available():
                device = f"{torch_device_type}:{torch_dev.current_device()}"
            else:
                device = "cpu:0"

        # The GPU buffer must reside on the same physical GPU that the
        # PhxCache's phxfs device maps to (e.g. GPU 4 -> phxfs_dev2).
        # phxfs_regmem performs a P2P mapping that fails with
        # "p2p_map->vaddrs _phxfs_regmem fail" (-14) when the GPU memory is
        # on a different (cross-NUMA) GPU than the phxfs device.
        dev = torch.device(device)
        if dev.type == "cuda":
            torch.cuda.set_device(dev)

        # phxfs_regmem requires 64K (GPU_PAGE_SIZE) alignment
        super().__init__(size, device, align_bytes=64 * 1024)

        self.phx_cache = phx_cache
        self.base_pointer = self.tensor.data_ptr()
        self.size = size

        # Register the entire pre-allocated GPU memory block with Phoenix.
        # target_base is the DMA-mapped address used by phxfs_read/read_async.
        self.target_base = phx_cache.regmem(self.base_pointer, size)

    def __del__(self) -> None:
        if hasattr(self, "base_pointer") and hasattr(self, "phx_cache"):
            try:
                self.phx_cache.deregmem(self.base_pointer, self.size)
            except Exception:
                pass

    def __str__(self) -> str:
        return "PhxFileMemoryAllocator"
