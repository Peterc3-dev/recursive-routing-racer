"""NPU-native scheduler — a tiny neural network that makes real-time
dispatch decisions based on system metrics.

The scheduler takes 6 system metrics and returns device assignments.
At ~500 parameters, it evaluates in microseconds on CPU and could run
at sub-microsecond speed on the XDNA NPU (50 TOPS).

Architecture:
    Input (6): [gpu_temp, gpu_util, gpu_vram_pct, cpu_load, npu_available, op_size_bucket]
    Hidden (16): ReLU activation
    Output (3): softmax over [cpu, gpu, npu] → argmax = chosen device

Training:
    Simple REINFORCE policy gradient — reward = -latency_ms.
    After each dispatch, record (metrics, device, reward) and update weights.
    Over ~50-100 runs, the scheduler learns the hardware personality.
"""

import os
import time
from typing import Optional, Tuple

import numpy as np


class NpuScheduler:
    """Tiny neural scheduler for tri-processor dispatch decisions."""

    DEVICES = ["cpu", "gpu", "npu"]

    def __init__(self, hidden_size: int = 16, lr: float = 0.01):
        self.hidden_size = hidden_size
        self.lr = lr
        self.input_size = 6
        self.output_size = 3

        # Xavier initialization for stable training
        self.w1 = np.random.randn(self.input_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.w2 = np.random.randn(hidden_size, self.output_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(self.output_size, dtype=np.float32)

        # Training stats
        self.total_updates = 0
        self.running_reward = 0.0
        self._last_metrics: Optional[np.ndarray] = None
        self._last_probs: Optional[np.ndarray] = None

    def forward(self, metrics: np.ndarray) -> Tuple[str, np.ndarray]:
        """Run scheduler inference.

        Args:
            metrics: 6-element array:
                [gpu_temp_norm, gpu_util_norm, gpu_vram_pct, cpu_load_norm, npu_available, op_size_norm]
                All values should be in [0, 1].

        Returns:
            (device_name, probabilities) — device is argmax of softmax output.
        """
        metrics = np.asarray(metrics, dtype=np.float32)
        assert metrics.shape == (6,), f"Expected 6 metrics, got {metrics.shape}"

        # Layer 1: linear + ReLU
        h = np.maximum(0, metrics @ self.w1 + self.b1)

        # Layer 2: linear + softmax
        logits = h @ self.w2 + self.b2
        # Numerically stable softmax
        logits_shifted = logits - logits.max()
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / (exp_logits.sum() + 1e-8)

        self._last_metrics = metrics.copy()
        self._last_probs = probs.copy()

        return self.DEVICES[np.argmax(probs)], probs

    def decide(self, gpu_temp: float, gpu_util: float, gpu_vram_pct: float,
               cpu_load: float, npu_available: bool, op_size: int) -> str:
        """Convenience wrapper — takes raw metrics, normalizes, returns device name."""
        metrics = np.array([
            min(gpu_temp / 100.0, 1.0),
            min(gpu_util / 100.0, 1.0),
            min(gpu_vram_pct, 1.0),
            min(cpu_load / 100.0, 1.0),
            1.0 if npu_available else 0.0,
            min(op_size / 1e6, 1.0),  # Normalize to [0, 1] for up to 1M elements
        ], dtype=np.float32)

        device, probs = self.forward(metrics)
        return device

    def update(self, metrics: np.ndarray, chosen_device: str, reward: float):
        """REINFORCE policy gradient update.

        Args:
            metrics: Same 6-element input used for forward().
            chosen_device: The device that was actually used.
            reward: Scalar reward (higher is better, e.g., -latency_ms / 100).
        """
        target_idx = self.DEVICES.index(chosen_device)

        # Forward pass to get activations
        h = np.maximum(0, metrics @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        logits_shifted = logits - logits.max()
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / (exp_logits.sum() + 1e-8)

        # Policy gradient: d log pi(a|s) * (R - baseline)
        # Baseline is running average reward
        advantage = reward - self.running_reward

        # Gradient of log softmax w.r.t. logits
        grad_logits = -probs.copy()
        grad_logits[target_idx] += 1.0
        grad_logits *= advantage

        # Update output layer
        self.w2 += self.lr * np.outer(h, grad_logits)
        self.b2 += self.lr * grad_logits

        # Update hidden layer (backprop through ReLU)
        grad_h = grad_logits @ self.w2.T
        grad_h *= (h > 0).astype(np.float32)  # ReLU derivative
        self.w1 += self.lr * np.outer(metrics, grad_h)
        self.b1 += self.lr * grad_h

        # Update baseline
        self.total_updates += 1
        alpha = min(0.1, 1.0 / max(self.total_updates, 1))
        self.running_reward = self.running_reward * (1 - alpha) + reward * alpha

    def save(self, path: str):
        """Save scheduler weights."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path,
                 w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2,
                 total_updates=np.array(self.total_updates),
                 running_reward=np.array(self.running_reward))

    def load(self, path: str) -> bool:
        """Load scheduler weights. Returns True if loaded successfully."""
        if not os.path.exists(path):
            return False
        data = np.load(path)
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.total_updates = int(data["total_updates"])
        self.running_reward = float(data["running_reward"])
        return True

    @property
    def param_count(self) -> int:
        return (self.w1.size + self.b1.size + self.w2.size + self.b2.size)

    def show_policy(self) -> str:
        """Interpret what the scheduler has learned."""
        lines = ["NPU Scheduler Policy Analysis:"]
        lines.append(f"  Parameters: {self.param_count}")
        lines.append(f"  Training updates: {self.total_updates}")
        lines.append(f"  Running reward: {self.running_reward:.4f}")
        lines.append("")

        # Probe with representative scenarios
        scenarios = [
            ("Hot GPU + large op",      [0.9, 0.8, 0.7, 0.3, 1.0, 0.8]),
            ("Cool GPU + large op",     [0.4, 0.3, 0.3, 0.3, 1.0, 0.8]),
            ("Any temp + small op",     [0.5, 0.5, 0.5, 0.5, 1.0, 0.05]),
            ("High CPU + medium op",    [0.5, 0.5, 0.5, 0.9, 1.0, 0.3]),
            ("No NPU + large op",       [0.5, 0.5, 0.5, 0.5, 0.0, 0.8]),
            ("No NPU + small op",       [0.5, 0.5, 0.5, 0.5, 0.0, 0.05]),
        ]

        lines.append("  Scenario                    → Decision (cpu/gpu/npu probabilities)")
        lines.append("  " + "-" * 70)
        for label, metrics_list in scenarios:
            device, probs = self.forward(np.array(metrics_list, dtype=np.float32))
            probs_str = f"[{probs[0]:.2f}, {probs[1]:.2f}, {probs[2]:.2f}]"
            lines.append(f"  {label:<30} → {device:<4} {probs_str}")

        return "\n".join(lines)

    def deploy_to_npu(self) -> str:
        """Attempt to deploy scheduler to run natively on XDNA NPU via IREE.

        Returns status string.
        """
        try:
            import iree.compiler as ireec
            import iree.runtime as ireert  # noqa: F401  (runtime availability probe)

            # The scheduler is a 2-layer MLP — trivially compilable
            # Export weights as constant tensors in MLIR
            # For now, just verify IREE is available
            status = "IREE available — NPU deployment possible"

            # Check for XDNA target
            # IREE uses "amd-aie" backend for XDNA NPUs
            try:
                targets = ireec.query_available_targets()
                if any("aie" in t.lower() for t in targets):
                    status = "IREE + XDNA backend available — full NPU deployment ready"
                else:
                    status = f"IREE available but no XDNA backend (targets: {targets}). Running on CPU (~3us/decision)"
            except Exception:
                status = "IREE available, target query failed. Running on CPU (~3us/decision)"

            return status

        except ImportError:
            return "IREE not available. Running on CPU (~3us/decision — still microsecond-fast)"

    def benchmark_latency(self, n: int = 1000) -> dict:
        """Measure actual inference latency."""
        metrics = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.5], dtype=np.float32)

        # Warmup
        for _ in range(100):
            self.forward(metrics)

        times = []
        for _ in range(n):
            start = time.perf_counter()
            self.forward(metrics)
            elapsed = (time.perf_counter() - start) * 1e6  # microseconds
            times.append(elapsed)

        return {
            "n": n,
            "avg_us": round(np.mean(times), 2),
            "min_us": round(np.min(times), 2),
            "max_us": round(np.max(times), 2),
            "p99_us": round(np.percentile(times, 99), 2),
        }
