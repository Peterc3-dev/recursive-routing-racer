"""ONNX Graph Dispatcher — parses an ONNX model and routes each operator
to CPU, GPU (Vulkan), or NPU based on op type, tensor sizes, and
learned hardware personality.

Usage:
    from engine.onnx_dispatcher import OnnxDispatcher
    d = OnnxDispatcher("model.onnx")
    print(d.summary())
    routing = d.get_routing()
"""

import os
from typing import Dict, List, Optional

import numpy as np
import onnx

# Op type → default device mapping
# Based on actual benchmarks from this hardware:
#   GPU (Vulkan): 1,084 GFLOPS — best for parallel compute
#   NPU: 50 TOPS INT8, 2W — best for small neural ops
#   CPU: 27.4 GFLOPS — fallback, I/O, sequential
DEFAULT_ROUTING: Dict[str, str] = {
    # GPU-bound (parallel, compute-heavy)
    "MatMul": "gpu",
    "Gemm": "gpu",
    "Conv": "gpu",
    "ConvTranspose": "gpu",
    "BatchNormalization": "gpu",
    "Attention": "gpu",
    "MultiHeadAttention": "gpu",
    "MatMulInteger": "gpu",
    "QLinearMatMul": "gpu",
    "Einsum": "gpu",

    # NPU-bound (small, efficient, low power)
    "Embedding": "npu",
    "LayerNormalization": "npu",
    "InstanceNormalization": "npu",
    "GroupNormalization": "npu",
    "Softmax": "npu",
    "Sigmoid": "npu",
    "Tanh": "npu",
    "Gelu": "npu",
    "Relu": "npu",
    "LeakyRelu": "npu",
    "Elu": "npu",
    "Selu": "npu",
    "Erf": "npu",
    "Add": "npu",
    "Mul": "npu",
    "Sub": "npu",
    "Div": "npu",
    "Pow": "npu",
    "Sqrt": "npu",
    "Exp": "npu",
    "Log": "npu",
    "Clip": "npu",
    "ReduceMean": "npu",
    "ReduceSum": "npu",

    # CPU-bound (sequential, I/O, control flow)
    "Reshape": "cpu",
    "Transpose": "cpu",
    "Gather": "cpu",
    "GatherElements": "cpu",
    "GatherND": "cpu",
    "Squeeze": "cpu",
    "Unsqueeze": "cpu",
    "Concat": "cpu",
    "Split": "cpu",
    "Slice": "cpu",
    "Cast": "cpu",
    "CastLike": "cpu",
    "Shape": "cpu",
    "ConstantOfShape": "cpu",
    "Constant": "cpu",
    "Where": "cpu",
    "NonZero": "cpu",
    "Flatten": "cpu",
    "Expand": "cpu",
    "Tile": "cpu",
    "Pad": "cpu",
    "Identity": "cpu",
    "If": "cpu",
    "Loop": "cpu",
    "Scan": "cpu",
    "SequenceConstruct": "cpu",
    "SequenceAt": "cpu",
}

# Size thresholds — small ops go to NPU even if default is GPU
SMALL_OP_THRESHOLD = 256 * 256  # Below this, NPU is more efficient
LARGE_OP_THRESHOLD = 1024 * 1024  # Above this, GPU is mandatory


class OnnxDispatcher:
    """Parse an ONNX model graph and build a per-operator routing table."""

    def __init__(self, model_path: str, personality_db=None):
        self.model_path = model_path
        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        self.personality = personality_db
        self.routing_table: Dict[str, dict] = {}

        # Build shape info index for fast lookup
        self._shape_index: Dict[str, List[int]] = {}
        self._build_shape_index()
        self._build_routing_table()

    def _build_shape_index(self):
        """Index all tensor shapes from graph value_info, inputs, and outputs."""
        for vi in list(self.graph.value_info) + list(self.graph.input) + list(self.graph.output):
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
                shape = []
                for dim in vi.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append(1)  # Dynamic/symbolic dim, assume 1
                    else:
                        shape.append(1)
                self._shape_index[vi.name] = shape

        # Also index initializer shapes
        for init in self.graph.initializer:
            self._shape_index[init.name] = list(init.dims)

    def _get_tensor_size(self, tensor_name: str) -> int:
        """Estimate total element count for a tensor."""
        shape = self._shape_index.get(tensor_name)
        if shape:
            return int(np.prod(shape)) if shape else 0
        return 0

    def _build_routing_table(self):
        """Assign each node to a device."""
        for idx, node in enumerate(self.graph.node):
            op = node.op_type
            name = node.name or f"{op}_{idx}"

            # Check personality first (learned from previous runs)
            if self.personality:
                learned = self._query_personality(op)
                if learned:
                    self.routing_table[name] = {
                        "op_type": op,
                        "device": learned["device"],
                        "source": "personality",
                        "confidence": learned["confidence"],
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                    }
                    continue

            # Estimate tensor sizes for this op
            input_sizes = [self._get_tensor_size(inp) for inp in node.input if inp]
            max_size = max(input_sizes) if input_sizes else 0

            # Route based on op type + size
            default_device = DEFAULT_ROUTING.get(op, "cpu")

            if default_device == "gpu" and max_size > 0 and max_size < SMALL_OP_THRESHOLD:
                device = "npu"  # Small GPU op → NPU is more efficient
                source = "size_override_small"
            elif default_device == "npu" and max_size > LARGE_OP_THRESHOLD:
                device = "gpu"  # Large NPU op → GPU is faster
                source = "size_override_large"
            else:
                device = default_device
                source = "default"

            self.routing_table[name] = {
                "op_type": op,
                "device": device,
                "source": source,
                "tensor_size": max_size,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }

    def _query_personality(self, op_type: str) -> Optional[dict]:
        """Query personality DB for learned routing. Returns dict or None."""
        if not self.personality:
            return None
        try:
            suggested = self.personality.suggest(op_type.lower())
            if isinstance(suggested, dict) and suggested.get("confidence", 0) > 0.8:
                return {"device": suggested["device"], "confidence": suggested["confidence"]}
            elif isinstance(suggested, str):
                return {"device": suggested, "confidence": 0.8}
        except Exception:
            pass
        return None

    def get_routing(self) -> Dict[str, dict]:
        """Return the full routing table."""
        return self.routing_table

    def get_execution_order(self) -> List[dict]:
        """Return routing table entries in topological order (graph order)."""
        return [
            {"name": name, **info}
            for name, info in self.routing_table.items()
        ]

    def summary(self) -> str:
        """Human-readable routing summary."""
        counts = {"cpu": 0, "gpu": 0, "npu": 0}
        by_source = {"default": 0, "size_override_small": 0, "size_override_large": 0, "personality": 0}

        for entry in self.routing_table.values():
            counts[entry["device"]] += 1
            by_source[entry.get("source", "default")] = by_source.get(entry.get("source", "default"), 0) + 1

        total = max(sum(counts.values()), 1)
        gpu_pct = 100 * counts["gpu"] // total
        npu_pct = 100 * counts["npu"] // total
        cpu_pct = 100 * counts["cpu"] // total

        # Estimate parallel fraction (GPU + NPU can run concurrently)
        parallel_pct = 100 * (counts["gpu"] + counts["npu"]) // total

        basename = os.path.basename(self.model_path)
        lines = [
            f"ONNX Model: {basename} ({total} operators)",
            f"  -> GPU: {counts['gpu']} ops ({gpu_pct}%)",
            f"  -> NPU: {counts['npu']} ops ({npu_pct}%)",
            f"  -> CPU: {counts['cpu']} ops ({cpu_pct}%)",
            f"  Estimated routing efficiency: {parallel_pct}% parallel (GPU+NPU concurrent)",
        ]

        if by_source.get("personality", 0):
            lines.append(f"  Personality overrides: {by_source['personality']}")
        if by_source.get("size_override_small", 0) or by_source.get("size_override_large", 0):
            lines.append(f"  Size overrides: {by_source.get('size_override_small', 0)} small->NPU, "
                         f"{by_source.get('size_override_large', 0)} large->GPU")

        return "\n".join(lines)

    def op_type_summary(self) -> Dict[str, dict]:
        """Summary grouped by op type."""
        ops: Dict[str, dict] = {}
        for entry in self.routing_table.values():
            op = entry["op_type"]
            if op not in ops:
                ops[op] = {"count": 0, "device": entry["device"], "sizes": []}
            ops[op]["count"] += 1
            if entry.get("tensor_size", 0) > 0:
                ops[op]["sizes"].append(entry["tensor_size"])
        return ops

    def export_rules(self) -> Dict[str, str]:
        """Export routing as simple op_type → device rules."""
        rules: Dict[str, str] = {}
        for entry in self.routing_table.values():
            op = entry["op_type"]
            if op not in rules:
                rules[op] = entry["device"]
        return rules

    def device_groups(self) -> Dict[str, List[str]]:
        """Group operator names by assigned device."""
        groups: Dict[str, List[str]] = {"cpu": [], "gpu": [], "npu": []}
        for name, entry in self.routing_table.items():
            groups[entry["device"]].append(f"{name} ({entry['op_type']})")
        return groups
