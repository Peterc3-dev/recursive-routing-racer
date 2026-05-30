"""Workload dispatcher — profiles operations per device, builds routing table.

After N runs, encodes dispatch patterns as lightweight rules. Re-evaluates
when exceptions occur (thermal spike, device error, shape mismatch).

Supports two dispatch modes:
  1. Heuristic/personality (original) — rule-based with learned personality
  2. Neural scheduler — tiny MLP trained via REINFORCE policy gradient

And ONNX model analysis:
  dispatcher.load_model("model.onnx") → per-operator routing table
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .personality import Personality
from .monitor import HardwareMonitor, SystemSnapshot
from .pulse import PulseController


class Device(Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"


@dataclass
class DispatchDecision:
    device: Device
    reason: str
    confidence: float = 0.0
    fallback: Optional[Device] = None


@dataclass
class OpProfile:
    """Runtime profile for an operation on a specific device."""
    device: Device
    operation: str
    avg_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    count: int = 0
    failures: int = 0
    last_temp: float = 0.0

    def update(self, duration_ms: float, temp: float = 0.0):
        self.count += 1
        self.last_temp = temp
        if self.count == 1:
            self.avg_ms = duration_ms
        else:
            self.avg_ms = self.avg_ms + (duration_ms - self.avg_ms) / self.count
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)


class Dispatcher:
    """Routes operations to the optimal device based on learned profiles."""

    LEARNING_THRESHOLD = 5  # Runs before encoding rules

    def __init__(
        self,
        monitor: HardwareMonitor,
        personality: Personality,
        pulse: PulseController,
        gpu_available: bool = True,
        npu_available: bool = False,
    ):
        self.monitor = monitor
        self.personality = personality
        self.pulse = pulse
        self.gpu_available = gpu_available
        self.npu_available = npu_available
        self._profiles: dict[tuple[str, str], OpProfile] = {}
        self._total_dispatches = 0
        self._reroutes = 0
        self._rules_encoded = False

    def dispatch(self, operation: str, input_size: int = 0) -> DispatchDecision:
        """Decide which device should handle this operation."""
        snap = self.monitor.snapshot
        self._total_dispatches += 1

        # Phase 1: Use personality rules only after thorough learning
        # Require rules_encoded AND enough dispatches (not OR) to avoid
        # locking into single-device routing before trying alternatives
        if self._rules_encoded and self._total_dispatches > self.LEARNING_THRESHOLD * 7:
            suggested = self.personality.suggest(
                operation, temp=snap.gpu.temp_c, input_size=input_size
            )
            device = Device(suggested)

            # Validate: is the suggested device actually usable right now?
            if device == Device.GPU and not self._gpu_usable(snap):
                self._reroutes += 1
                device = Device.NPU if self.npu_available else Device.CPU
                return DispatchDecision(
                    device=device,
                    reason=f"reroute: gpu thermal → {device.value}",
                    confidence=0.7,
                    fallback=Device.CPU,
                )
            if device == Device.NPU and not self.npu_available:
                self._reroutes += 1
                device = Device.CPU
                return DispatchDecision(
                    device=device,
                    reason="reroute: npu unavailable → cpu",
                    confidence=0.7,
                )

            return DispatchDecision(
                device=device,
                reason=f"personality rule: {operation} → {device.value}",
                confidence=0.8,
            )

        # Phase 0: Heuristic routing during learning period
        return self._heuristic_dispatch(operation, input_size, snap)

    def _heuristic_dispatch(
        self, operation: str, input_size: int, snap: SystemSnapshot
    ) -> DispatchDecision:
        """Simple heuristics before enough data for learned rules."""

        # Compute-heavy matrix ops → GPU if available and cool enough
        if operation in ("matmul", "conv2d", "attention", "gemm", "project"):
            if self.gpu_available and self._gpu_usable(snap):
                return DispatchDecision(
                    device=Device.GPU,
                    reason=f"heuristic: {operation} → gpu (compute-heavy)",
                    confidence=0.5,
                    fallback=Device.CPU,
                )

        # NPU-suitable ops: embeddings, normalization, quantized, LLM generation
        if self.npu_available and operation in (
            "embed", "normalize", "layernorm", "rmsnorm",
            "quantized_matmul", "int8_gemm", "llm_generate",
            "npu_inference", "token_generation", "embedding",
        ):
            return DispatchDecision(
                device=Device.NPU,
                reason=f"heuristic: {operation} → npu (efficient)",
                confidence=0.5,
                fallback=Device.CPU,
            )

        # Lightweight ops → CPU
        if operation in ("softmax", "relu", "tokenize", "decode"):
            return DispatchDecision(
                device=Device.CPU,
                reason=f"heuristic: {operation} → cpu (lightweight)",
                confidence=0.6,
            )

        # Default to CPU
        return DispatchDecision(
            device=Device.CPU,
            reason="heuristic: default → cpu",
            confidence=0.3,
        )

    def _gpu_usable(self, snap: SystemSnapshot) -> bool:
        """Check if GPU is available and within thermal budget."""
        if not self.gpu_available:
            return False
        if not self.pulse.should_fire_gpu(snap.gpu.temp_c):
            return False
        return True

    def record_result(
        self,
        device: Device,
        operation: str,
        duration_ms: float,
        input_size: int = 0,
        success: bool = True,
    ):
        """Record execution result to improve future dispatching."""
        snap = self.monitor.snapshot
        key = (device.value, operation)

        if key not in self._profiles:
            self._profiles[key] = OpProfile(device=device, operation=operation)
        self._profiles[key].update(duration_ms, snap.gpu.temp_c)

        if not success:
            self._profiles[key].failures += 1

        self.personality.record_run(
            device=device.value,
            operation=operation,
            duration_ms=duration_ms,
            input_size=input_size,
            temp_before=snap.gpu.temp_c,
            power_w=snap.gpu.power_w,
            success=success,
        )

        # Check if we should encode rules
        total_profile_runs = sum(p.count for p in self._profiles.values())
        if not self._rules_encoded and total_profile_runs >= self.LEARNING_THRESHOLD * 3:
            self.personality.update_rules()
            self._rules_encoded = True

    def force_reencode(self):
        """Force re-evaluation of routing rules (e.g., after driver update)."""
        self.personality.update_rules()
        self._rules_encoded = True

    # ------------------------------------------------------------------
    # ONNX model analysis
    # ------------------------------------------------------------------

    def load_model(self, model_path: str) -> str:
        """Load an ONNX model and build per-operator routing from its graph.

        Returns the routing summary string.
        """
        from .onnx_dispatcher import OnnxDispatcher

        self._onnx_dispatcher = OnnxDispatcher(model_path, self.personality)
        self._onnx_routing = self._onnx_dispatcher.get_routing()
        self._onnx_rules = self._onnx_dispatcher.export_rules()
        return self._onnx_dispatcher.summary()

    def get_onnx_routing(self) -> dict:
        """Return ONNX model routing table (after load_model)."""
        if hasattr(self, "_onnx_routing"):
            return self._onnx_routing
        return {}

    # ------------------------------------------------------------------
    # Neural scheduler integration
    # ------------------------------------------------------------------

    def enable_neural_scheduler(self, weights_path: str = None):
        """Switch to the neural (NpuScheduler) dispatch mode.

        The scheduler is a tiny MLP trained via REINFORCE — it replaces
        the heuristic dispatch with a learned policy.
        """
        from .npu_scheduler import NpuScheduler

        self._neural_scheduler = NpuScheduler()
        self._scheduler_path = weights_path or os.path.expanduser(
            "~/.rag-race-router/scheduler.npz"
        )
        self._neural_scheduler.load(self._scheduler_path)
        self._use_neural = True

    def neural_dispatch(self, operation: str, input_size: int = 0) -> DispatchDecision:
        """Dispatch using the neural scheduler."""
        if not hasattr(self, "_neural_scheduler") or not self._use_neural:
            return self.dispatch(operation, input_size)

        snap = self.monitor.snapshot

        metrics = np.array([
            min(snap.gpu.temp_c / 100.0, 1.0),
            min(snap.gpu.util_pct / 100.0, 1.0),
            min(snap.gpu.vram_used_mb / max(snap.gpu.vram_total_mb, 1) , 1.0),
            min(snap.cpu.load_avg[0] / 100.0, 1.0) if snap.cpu.load_avg else 0.5,
            1.0 if self.npu_available else 0.0,
            min(input_size / 1e6, 1.0),
        ], dtype=np.float32)

        device_name, probs = self._neural_scheduler.forward(metrics)
        self._last_neural_metrics = metrics

        device = Device(device_name)

        # Validate usability
        if device == Device.GPU and not self._gpu_usable(snap):
            device = Device.NPU if self.npu_available else Device.CPU
        if device == Device.NPU and not self.npu_available:
            device = Device.CPU

        return DispatchDecision(
            device=device,
            reason=f"neural: {operation} -> {device.value} (p={probs[self._neural_scheduler.DEVICES.index(device.value)]:.2f})",
            confidence=float(probs[self._neural_scheduler.DEVICES.index(device.value)]),
            fallback=Device.CPU,
        )

    def record_neural_result(self, device_str: str, latency_ms: float):
        """Feed result back to neural scheduler for learning."""
        if hasattr(self, "_neural_scheduler") and hasattr(self, "_last_neural_metrics"):
            reward = -latency_ms / 100.0  # Lower latency = higher reward
            self._neural_scheduler.update(self._last_neural_metrics, device_str, reward)

    def save_scheduler(self):
        """Persist scheduler weights."""
        if hasattr(self, "_neural_scheduler"):
            self._neural_scheduler.save(self._scheduler_path)

    @property
    def stats(self) -> dict:
        base = {
            "total_dispatches": self._total_dispatches,
            "reroutes": self._reroutes,
            "rules_encoded": self._rules_encoded,
            "profiles": {
                f"{k[0]}/{k[1]}": {
                    "avg_ms": round(p.avg_ms, 2),
                    "count": p.count,
                    "failures": p.failures,
                }
                for k, p in self._profiles.items()
            },
        }
        if hasattr(self, "_neural_scheduler"):
            base["neural_scheduler"] = {
                "param_count": self._neural_scheduler.param_count,
                "training_updates": self._neural_scheduler.total_updates,
                "running_reward": round(self._neural_scheduler.running_reward, 4),
            }
        if hasattr(self, "_onnx_dispatcher"):
            base["onnx_model"] = self._onnx_dispatcher.summary()
        return base
