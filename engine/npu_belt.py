"""NPU execution belt — persistent XRT runtime with Triton-XDNA compiled kernels.

No IREE. Triton-XDNA compiles kernels straight to xclbin, XRT dispatches
directly to XDNA2 AIE tiles. Hardware context stays warm between calls.
Pre-allocated buffer objects eliminate per-call allocation overhead.

Proven dispatch: ~500us with persistent runtime vs ~62ms cold.

Backend priority:
  1. Persistent XRT (pre-allocated BOs, warm hw context)
  2. CPU fallback (NumPy)
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

@dataclass
class OpLatency:
    """Per-op latency stats for personality DB integration."""
    op_name: str
    backend: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    last_ms: float = 0.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0

    def record(self, duration_ms: float):
        self.count += 1
        self.total_ms += duration_ms
        self.last_ms = duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)

    def to_dict(self) -> dict:
        return {
            "op": self.op_name,
            "backend": self.backend,
            "count": self.count,
            "avg_ms": round(self.avg_ms, 3),
            "min_ms": round(self.min_ms, 3) if self.min_ms != float("inf") else 0.0,
            "max_ms": round(self.max_ms, 3),
            "last_ms": round(self.last_ms, 3),
        }


# ---------------------------------------------------------------------------
# Persistent XRT runtime (the proven path)
# ---------------------------------------------------------------------------

class PersistentXrtBackend:
    """Persistent XRT runtime — load xclbin ONCE, dispatch many.

    Keeps hardware context warm and buffer objects pre-allocated.
    Based on persistent_npu_v2.py which proved ~500us dispatch.

    Requires:
      - /dev/accel/accel0 (XDNA driver loaded)
      - pyxrt bindings
      - Pre-compiled xclbin + insts.bin from Triton-XDNA
    """

    # Default paths for Triton-XDNA compiled matmul kernel
    DEFAULT_XCLBIN = "/tmp/Triton-XDNA/examples/air_project/aie.xclbin"
    DEFAULT_INSTS = "/tmp/Triton-XDNA/examples/air_project/insts.bin"

    def __init__(self, xclbin_path=None, insts_path=None, M=256, K=256, N=256):
        self._available = False
        self._pyxrt = None
        self._device = None
        self._hw_ctx = None
        self._kernel = None
        self._insts_bo = None
        self._insts_len = 0
        self._buffers = {}  # Pre-allocated BOs keyed by (M, K, N)
        self._xclbin_path = xclbin_path or self.DEFAULT_XCLBIN
        self._insts_path = insts_path or self.DEFAULT_INSTS
        self._default_dims = (M, K, N)
        self._probe()

    def _probe(self):
        """Detect XRT + XDNA hardware and initialize persistent context."""
        if not Path("/dev/accel/accel0").exists():
            return

        try:
            import pyxrt
            self._pyxrt = pyxrt
        except ImportError:
            return

        if not Path(self._xclbin_path).exists():
            return

        try:
            self._init_persistent_context()
            self._available = True
        except Exception as e:
            print(f"[NPU Belt] XRT init failed: {e}")

    def _init_persistent_context(self):
        """One-time setup: device, xclbin, hw context, kernel, instruction BO."""
        pyxrt = self._pyxrt

        self._device = pyxrt.device(0)
        xclbin = pyxrt.xclbin(self._xclbin_path)
        self._device.register_xclbin(xclbin)
        self._hw_ctx = pyxrt.hw_context(self._device, xclbin.get_uuid())

        xkernels = xclbin.get_kernels()
        self._kernel = pyxrt.kernel(self._hw_ctx, xkernels[0].get_name())

        # Load and upload instruction sequence (once)
        with open(self._insts_path, 'rb') as f:
            insts_data = f.read()
        self._insts_len = len(insts_data) // 4

        self._insts_bo = pyxrt.bo(
            self._device, len(insts_data),
            pyxrt.bo.flags.cacheable, self._kernel.group_id(1)
        )
        self._insts_bo.write(insts_data, 0)
        self._insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Pre-allocate default size buffers
        self._preallocate(*self._default_dims)

    def _preallocate(self, M, K, N):
        """Pre-allocate input/output buffer objects for given dimensions."""
        pyxrt = self._pyxrt
        key = (M, K, N)
        if key in self._buffers:
            return

        a_size = M * K * 2  # bf16
        b_size = K * N * 2  # bf16
        c_size = M * N * 4  # f32

        a_bo = pyxrt.bo(self._device, a_size,
                        pyxrt.bo.flags.cacheable, self._kernel.group_id(3))
        b_bo = pyxrt.bo(self._device, b_size,
                        pyxrt.bo.flags.cacheable, self._kernel.group_id(4))
        c_bo = pyxrt.bo(self._device, c_size,
                        pyxrt.bo.flags.cacheable, self._kernel.group_id(5))

        self._buffers[key] = (a_bo, b_bo, c_bo)

    @property
    def available(self) -> bool:
        return self._available

    def execute_matmul(self, a_bytes: bytes, b_bytes: bytes,
                       M=256, K=256, N=256) -> bytes:
        """Dispatch matmul — only memcpy, no allocation."""
        if not self._available:
            return None

        key = (M, K, N)
        if key not in self._buffers:
            self._preallocate(M, K, N)

        pyxrt = self._pyxrt
        a_bo, b_bo, c_bo = self._buffers[key]

        # Write data to pre-allocated BOs
        a_bo.write(a_bytes, 0)
        b_bo.write(b_bytes, 0)
        a_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        b_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Dispatch (kernel handle already warm)
        run = self._kernel(3, self._insts_bo, self._insts_len,
                           a_bo, b_bo, c_bo)
        run.wait()

        # Read result
        c_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return c_bo.read(M * N * 4, 0)

    def dispatch_only(self):
        """Dispatch with existing data — measure pure kernel overhead."""
        if not self._available:
            return
        key = self._default_dims
        if key not in self._buffers:
            return
        a_bo, b_bo, c_bo = self._buffers[key]
        run = self._kernel(3, self._insts_bo, self._insts_len,
                           a_bo, b_bo, c_bo)
        run.wait()

    def status(self) -> dict:
        return {
            "available": self._available,
            "xclbin": self._xclbin_path,
            "preallocated_dims": list(self._buffers.keys()),
            "hw_context": "warm" if self._hw_ctx else "none",
        }


# ---------------------------------------------------------------------------
# CPU fallback ops
# ---------------------------------------------------------------------------

def _cpu_attention(q, k, v):
    d_k = q.shape[-1]
    scores = (q @ k.T) / np.sqrt(d_k)
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v


def _cpu_softmax(x):
    x_shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


_CPU_OPS: Dict[str, Callable] = {
    "matmul": lambda *t: t[0] @ t[1] if len(t) == 2 else None,
    "normalize": lambda *t: t[0] / np.sqrt(np.mean(t[0]**2, axis=-1, keepdims=True) + 1e-6),
    "rmsnorm": lambda *t: t[0] / np.sqrt(np.mean(t[0]**2, axis=-1, keepdims=True) + 1e-6),
    "attention": lambda *t: _cpu_attention(*t) if len(t) == 3 else None,
    "softmax": lambda *t: _cpu_softmax(t[0]) if len(t) == 1 else None,
}


# ---------------------------------------------------------------------------
# Unified NPU Execution Belt
# ---------------------------------------------------------------------------

class NpuExecutionBelt:
    """NPU dispatch belt — persistent XRT with CPU fallback.

    No IREE. Triton-XDNA compiles to xclbin, XRT dispatches directly.

    Usage:
        belt = NpuExecutionBelt()
        result = belt.dispatch("matmul", a, b)
    """

    def __init__(self, personality=None, xclbin_path=None, insts_path=None):
        self._xrt = PersistentXrtBackend(
            xclbin_path=xclbin_path, insts_path=insts_path
        )
        self._personality = personality
        self._latency: Dict[str, OpLatency] = {}

    @property
    def active_backend(self) -> str:
        if self._xrt.available:
            return "persistent-xrt"
        return "cpu_fallback"

    @property
    def npu_available(self) -> bool:
        return self._xrt.available

    def dispatch(self, op_name: str, *tensors: np.ndarray) -> np.ndarray:
        """Dispatch to persistent XRT or CPU fallback."""
        start = time.perf_counter()
        result = None
        backend_used = "cpu_fallback"

        # Try persistent XRT for matmul (the proven kernel)
        if self._xrt.available and op_name == "matmul" and len(tensors) == 2:
            a, b = tensors
            a_bf16 = a.astype(np.float16).tobytes()
            b_bf16 = b.astype(np.float16).tobytes()
            M, K = a.shape
            _, N = b.shape
            c_bytes = self._xrt.execute_matmul(a_bf16, b_bf16, M, K, N)
            if c_bytes is not None:
                result = np.frombuffer(bytes(c_bytes), dtype=np.float32).reshape(M, N)
                backend_used = "persistent-xrt"

        # CPU fallback for everything else (or if XRT dispatch failed)
        if result is None:
            cpu_fn = _CPU_OPS.get(op_name)
            if cpu_fn is not None:
                result = cpu_fn(*tensors)
                backend_used = "cpu_fallback"

        if result is None:
            raise ValueError(f"Unsupported op: {op_name}")

        # Record latency
        elapsed_ms = (time.perf_counter() - start) * 1000
        key = f"{op_name}:{backend_used}"
        if key not in self._latency:
            self._latency[key] = OpLatency(op_name=op_name, backend=backend_used)
        self._latency[key].record(elapsed_ms)

        # Feed personality DB
        if self._personality is not None:
            self._personality.record_run(
                device="npu" if backend_used != "cpu_fallback" else "cpu",
                operation=op_name,
                duration_ms=elapsed_ms,
                input_size=sum(t.size for t in tensors),
                success=True,
                metadata={"npu_backend": backend_used},
            )

        return result

    @property
    def latency_stats(self) -> Dict[str, dict]:
        return {k: v.to_dict() for k, v in self._latency.items()}

    def status(self) -> dict:
        return {
            "active_backend": self.active_backend,
            "npu_available": self.npu_available,
            "xrt": self._xrt.status(),
            "latency": self.latency_stats,
        }

    def __repr__(self) -> str:
        return f"NpuExecutionBelt(backend={self.active_backend!r})"
