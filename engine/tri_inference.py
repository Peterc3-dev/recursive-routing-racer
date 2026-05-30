"""Tri-processor inference bridge — HDC routes each op to CPU/GPU/NPU.

Connects:
  - unified-ml GGUF loader (weights)
  - torch-vulkan (GPU belt)
  - persistent XRT (NPU belt)
  - hdc_scheduler (routing brain)
"""

import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TriProcessorInference:
    """Run model layers through HDC-routed tri-processor dispatch.

    Each operation gets routed independently based on:
    - Matrix shape (M, K, N)
    - GPU temperature
    - CPU load
    - Operation type (matmul, attention, gelu, layernorm, etc.)
    """

    def __init__(self, hdc_scheduler=None, monitor=None):
        self._hdc = hdc_scheduler
        self._monitor = monitor

        # GPU belt (torch-vulkan)
        import importlib.util
        self._gpu_available = importlib.util.find_spec("torch_vulkan") is not None

        # NPU belt (persistent XRT)
        self._npu_belt = None
        try:
            from engine.npu_belt import NpuExecutionBelt
            self._npu_belt = NpuExecutionBelt()
        except (ImportError, Exception):
            pass

        self._stats = {
            'cpu': {'count': 0, 'total_ms': 0},
            'gpu': {'count': 0, 'total_ms': 0},
            'npu': {'count': 0, 'total_ms': 0},
        }

    def _get_metrics(self, op_type, tensor_a=None, tensor_b=None):
        """Build metrics dict for HDC encoding."""
        metrics = {
            'op_type': op_type,
            'op_size': 0,
            'gpu_temp': 50,
            'gpu_util': 0,
            'cpu_load': 0,
            'npu_available': self._npu_belt is not None and self._npu_belt.npu_available,
        }

        if tensor_a is not None:
            metrics['op_size'] = tensor_a.numel() * 4  # bytes

        # Live telemetry if monitor available
        if self._monitor:
            try:
                hw = self._monitor.snapshot()
                metrics['gpu_temp'] = hw.get('gpu_temp', 50)
                metrics['gpu_util'] = hw.get('gpu_util', 0)
                metrics['cpu_load'] = hw.get('cpu_load', 0)
            except Exception:
                pass

        return metrics

    def route(self, op_type, *tensors):
        """Ask HDC where to run this op, then execute there."""
        a = tensors[0] if tensors else None
        b = tensors[1] if len(tensors) > 1 else None
        metrics = self._get_metrics(op_type, a, b)

        # HDC routing decision
        if self._hdc and self._hdc.codebook:
            device, _confidence = self._hdc.dispatch(metrics)
        else:
            # Cold start: shape-based heuristic
            device = self._heuristic_route(op_type, a, b)

        # Execute on chosen device
        t0 = time.perf_counter()
        result = self._execute(device, op_type, *tensors)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Record outcome for HDC learning
        if self._hdc:
            self._hdc.record(metrics, device, elapsed_ms)

        # Track stats
        self._stats[device]['count'] += 1
        self._stats[device]['total_ms'] += elapsed_ms

        return result, device, elapsed_ms

    def _heuristic_route(self, op_type, a=None, b=None):
        """Shape-based routing when codebook is empty."""
        if a is None:
            return 'cpu'

        numel = a.numel()

        if op_type == 'matmul' and b is not None:
            # Small matmuls: CPU. Large: GPU.
            total_ops = a.shape[0] * a.shape[1] * b.shape[1]
            if total_ops < 1_000_000:  # < 1M ops
                return 'cpu'
            elif self._gpu_available:
                return 'gpu'

        if op_type in ('gelu', 'relu', 'layernorm', 'softmax'):
            if numel < 100_000:
                return 'cpu'
            elif self._gpu_available:
                return 'gpu'

        if op_type == 'attention' and self._gpu_available:
            return 'gpu'

        return 'cpu'

    def _execute(self, device, op_type, *tensors):
        """Execute op on the chosen device."""
        if device == 'gpu' and self._gpu_available:
            return self._execute_gpu(op_type, *tensors)
        elif device == 'npu' and self._npu_belt and self._npu_belt.npu_available:
            return self._execute_npu(op_type, *tensors)
        else:
            return self._execute_cpu(op_type, *tensors)

    def _execute_cpu(self, op_type, *tensors):
        """CPU execution via PyTorch."""
        ts = [t.cpu() if hasattr(t, 'cpu') else t for t in tensors]
        if op_type == 'matmul':
            return torch.mm(ts[0], ts[1])
        elif op_type == 'addmm':
            return torch.addmm(ts[0], ts[1], ts[2])
        elif op_type == 'gelu':
            return torch.nn.functional.gelu(ts[0])
        elif op_type == 'layernorm':
            return torch.nn.functional.layer_norm(ts[0], [ts[0].shape[-1]], ts[1], ts[2])
        elif op_type == 'attention':
            return torch.nn.functional.scaled_dot_product_attention(ts[0], ts[1], ts[2])
        return ts[0]

    def _execute_gpu(self, op_type, *tensors):
        """GPU execution via torch-vulkan."""
        ts = [t.to('vulkan') if t.device.type != 'privateuseone' else t for t in tensors]
        if op_type == 'matmul':
            return torch.mm(ts[0], ts[1])
        elif op_type == 'addmm':
            return torch.addmm(ts[0], ts[1], ts[2])
        elif op_type == 'gelu':
            return torch.nn.functional.gelu(ts[0])
        elif op_type == 'layernorm':
            return torch.nn.functional.layer_norm(ts[0], [ts[0].shape[-1]], ts[1], ts[2])
        elif op_type == 'attention':
            return torch.nn.functional.scaled_dot_product_attention(ts[0], ts[1], ts[2])
        return ts[0]

    def _execute_npu(self, op_type, *tensors):
        """NPU execution via persistent XRT belt."""
        ts = [t.cpu().numpy() if hasattr(t, 'numpy') else t for t in tensors]
        result_np = self._npu_belt.dispatch(op_type, *ts)
        return torch.from_numpy(result_np)

    @property
    def stats(self):
        lines = ["Tri-Processor Routing Stats:"]
        for dev in ('cpu', 'gpu', 'npu'):
            s = self._stats[dev]
            if s['count'] > 0:
                avg = s['total_ms'] / s['count']
                lines.append(f"  {dev}: {s['count']} ops, {avg:.2f}ms avg, {s['total_ms']:.0f}ms total")
        return "\n".join(lines)
