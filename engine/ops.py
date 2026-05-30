"""Concrete operation implementations for CPU, GPU (Vulkan/Kompute), and NPU.

Each function returns its result and is designed to be submitted as a WorkItem.fn
to the appropriate belt via the dispatcher.
"""

import struct
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Vulkan SPIR-V shader paths
# ---------------------------------------------------------------------------
_SHADER_DIR = Path.home() / "projects" / "unified-ml" / "vulkan"
_MATMUL_SPV = _SHADER_DIR / "matmul.spv"
_ATTENTION_SPV = _SHADER_DIR / "attention.spv"
_FUSED_SPV = _SHADER_DIR / "unified_memory.spv"

# Lazy-loaded Kompute manager (singleton)
_kp_mgr = None


def _get_kp_manager():
    global _kp_mgr
    if _kp_mgr is None:
        import kp
        _kp_mgr = kp.Manager()
    return _kp_mgr


# ---------------------------------------------------------------------------
# CPU Operations
# ---------------------------------------------------------------------------

def cpu_tokenize(text: str) -> np.ndarray:
    """Simple whitespace tokenizer → integer token IDs."""
    words = text.lower().split()
    # Deterministic hash-based token IDs (no external tokenizer needed)
    tokens = np.array([hash(w) % 32000 for w in words], dtype=np.int32)
    return tokens


def cpu_decode(logits: np.ndarray) -> str:
    """Greedy decode: argmax over logits → token string."""
    token_ids = np.argmax(logits, axis=-1)
    # Dummy decode — just return token IDs as string
    return " ".join(str(t) for t in token_ids.flatten()[:10])


def cpu_normalize(tensor: np.ndarray) -> np.ndarray:
    """RMS normalization (CPU fallback)."""
    rms = np.sqrt(np.mean(tensor ** 2, axis=-1, keepdims=True) + 1e-6)
    return tensor / rms


def cpu_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply on CPU (NumPy/BLAS)."""
    return a @ b


def cpu_embed(tokens: np.ndarray, dim: int = 512) -> np.ndarray:
    """Simple embedding lookup (deterministic, seeded by token IDs)."""
    rng = np.random.RandomState(42)
    table = rng.randn(32000, dim).astype(np.float32) * 0.02
    return table[tokens % 32000]


def cpu_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention on CPU."""
    d_k = q.shape[-1]
    scores = (q @ k.T) / np.sqrt(d_k)
    # Stable softmax
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ v


def cpu_project(tensor: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Linear projection on CPU."""
    return tensor @ weight


# ---------------------------------------------------------------------------
# GPU Operations (Vulkan / Kompute)
# ---------------------------------------------------------------------------

def gpu_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply on GPU via Vulkan/Kompute SPIR-V shader."""
    import kp

    mgr = _get_kp_manager()
    n = a.shape[0]
    assert a.shape == (n, n) and b.shape == (n, n), "Square matrices required"

    spirv = _MATMUL_SPV.read_bytes()

    t_a = mgr.tensor(a.flatten().tolist())
    t_b = mgr.tensor(b.flatten().tolist())
    t_c = mgr.tensor([0.0] * (n * n))
    params = [t_a, t_b, t_c]

    push = [struct.unpack("f", struct.pack("I", n))[0]]
    wg = max(1, n // 16)

    seq = mgr.sequence()
    seq.record(kp.OpSyncDevice(params))
    algo = mgr.algorithm(params, spirv, [wg, wg, 1], [], push)
    seq.record(kp.OpAlgoDispatch(algo))
    seq.record(kp.OpSyncLocal([t_c]))
    seq.eval()

    return np.array(t_c.data()).reshape(n, n).astype(np.float32)


def gpu_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Scaled dot-product attention on GPU via Vulkan/Kompute."""
    import kp

    mgr = _get_kp_manager()
    seq_len = q.shape[0]
    head_dim = q.shape[1]

    if not _ATTENTION_SPV.exists():
        # Fallback to CPU attention if shader not available
        return cpu_attention(q, k, v)

    spirv = _ATTENTION_SPV.read_bytes()

    # Pack Q, K, V as flat float arrays
    t_q = mgr.tensor(q.flatten().tolist())
    t_k = mgr.tensor(k.flatten().tolist())
    t_v = mgr.tensor(v.flatten().tolist())
    t_out = mgr.tensor([0.0] * (seq_len * head_dim))
    params = [t_q, t_k, t_v, t_out]

    push = [
        struct.unpack("f", struct.pack("I", seq_len))[0],
        struct.unpack("f", struct.pack("I", head_dim))[0],
    ]

    seq = mgr.sequence()
    seq.record(kp.OpSyncDevice(params))
    algo = mgr.algorithm(params, spirv, [seq_len, 1, 1], [], push)
    seq.record(kp.OpAlgoDispatch(algo))
    seq.record(kp.OpSyncLocal([t_out]))
    seq.eval()

    return np.array(t_out.data()).reshape(seq_len, head_dim).astype(np.float32)


def gpu_project(tensor: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Linear projection via GPU matmul."""
    # Pad to square if needed for the matmul shader, or use CPU for non-square
    m, k = tensor.shape
    k2, n = weight.shape
    if k != k2:
        return cpu_project(tensor, weight)

    # For the demo, use GPU matmul if dimensions allow, else CPU fallback
    if m == k and k == n:
        return gpu_matmul(tensor, weight)
    # Non-square: use CPU
    return cpu_project(tensor, weight)


# ---------------------------------------------------------------------------
# NPU Operations (via FLM or direct)
# ---------------------------------------------------------------------------

def npu_embed(tokens: np.ndarray, dim: int = 512) -> np.ndarray:
    """Embedding on NPU — for demo, uses optimized CPU path.

    Real NPU embedding would use IREE or FLM's internal embedding layer.
    For the dispatch demo, this simulates NPU execution with the same
    interface so the routing logic is exercised.
    """
    return cpu_embed(tokens, dim)


def npu_normalize(tensor: np.ndarray) -> np.ndarray:
    """RMS normalization — dispatched to NPU belt.

    The NPU excels at element-wise ops on quantized data. For the demo,
    this uses the same math but runs through the NPU belt's thread.
    """
    return cpu_normalize(tensor)


# ---------------------------------------------------------------------------
# Demo Workload Builder
# ---------------------------------------------------------------------------

def build_demo_workload(seq_len: int = 16, dim: int = 512, proj_dim: int = 128):
    """Build a demo inference pipeline workload.

    Returns a list of (operation_name, fn, args, kwargs, input_size, deps) tuples.
    deps is a list of operation names this op depends on.
    """
    text = "The quick brown fox jumps over the lazy dog"
    np.random.seed(42)

    # Pre-generate data for independent ops
    a = np.random.randn(dim, dim).astype(np.float32)
    b = np.random.randn(dim, dim).astype(np.float32)

    workload = [
        # (name, cpu_fn, gpu_fn, npu_fn, args, input_size, deps)
        {
            "name": "tokenize",
            "cpu": (cpu_tokenize, (text,)),
            "gpu": None,  # CPU-only
            "npu": None,
            "input_size": len(text),
            "deps": [],
        },
        {
            "name": "embed",
            "cpu": (cpu_embed, None),  # args filled from tokenize result
            "gpu": None,
            "npu": (npu_embed, None),
            "input_size": seq_len * dim,
            "deps": ["tokenize"],
        },
        {
            "name": "matmul",
            "cpu": (cpu_matmul, (a, b)),
            "gpu": (gpu_matmul, (a, b)),
            "npu": None,
            "input_size": dim * dim,
            "deps": [],
        },
        {
            "name": "attention",
            "cpu": (cpu_attention, None),  # args from embed
            "gpu": (gpu_attention, None),
            "npu": None,
            "input_size": seq_len * dim,
            "deps": ["embed"],
        },
        {
            "name": "normalize",
            "cpu": (cpu_normalize, None),  # args from attention
            "gpu": None,
            "npu": (npu_normalize, None),
            "input_size": seq_len * dim,
            "deps": ["attention"],
        },
        {
            "name": "project",
            "cpu": (cpu_project, None),  # args from normalize
            "gpu": (gpu_project, None),
            "npu": None,
            "input_size": seq_len * proj_dim,
            "deps": ["normalize"],
        },
        {
            "name": "decode",
            "cpu": (cpu_decode, None),  # args from project
            "gpu": None,
            "npu": None,
            "input_size": proj_dim,
            "deps": ["project"],
        },
    ]

    return workload
