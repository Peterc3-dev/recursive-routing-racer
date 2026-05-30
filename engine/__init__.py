"""R.A.G-Race-Router -- Adaptive Tri-Processor Inference Runtime.

Self-optimizing inference engine that coordinates CPU, iGPU (Vulkan), and NPU
on AMD Ryzen AI 300 series APUs.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .monitor import HardwareMonitor
from .dispatcher import Device, Dispatcher
from .executor import Executor, WorkItem, Fragment
from .personality import Personality
from .pulse import PulseConfig, PulseController

__all__ = [
    "EngineConfig",
    "RagRaceRouter",
    "HardwareMonitor",
    "Device",
    "Dispatcher",
    "Executor",
    "WorkItem",
    "Fragment",
    "Personality",
    "PulseConfig",
    "PulseController",
]


@dataclass
class EngineConfig:
    """Engine configuration."""
    gpu_burst_ms: float = 50.0
    cooldown_ms: float = 10.0
    temp_ceiling: float = 80.0
    temp_floor: float = 60.0
    monitor_interval_ms: int = 500
    cpu_workers: int = 4
    gpu_enabled: bool = True
    npu_enabled: bool = True
    db_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "EngineConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_file(cls, path: str) -> "EngineConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict:
        return {
            k: getattr(self, k) for k in self.__dataclass_fields__
        }


class RagRaceRouter:
    """Main engine -- coordinates monitor, dispatcher, executor, personality."""

    def __init__(self, config: EngineConfig = None):
        self.config = config or EngineConfig()

        # Initialize components
        self.monitor = HardwareMonitor(interval_ms=self.config.monitor_interval_ms)

        db_path = Path(self.config.db_path) if self.config.db_path else None
        self.personality = Personality(db_path=db_path)

        self.pulse = PulseController(PulseConfig(
            gpu_burst_ms=self.config.gpu_burst_ms,
            cooldown_ms=self.config.cooldown_ms,
            temp_ceiling=self.config.temp_ceiling,
            temp_floor=self.config.temp_floor,
        ))

        # Detect hardware
        npu_available = self.config.npu_enabled and Path("/dev/accel/accel0").exists()
        self.dispatcher = Dispatcher(
            monitor=self.monitor,
            personality=self.personality,
            pulse=self.pulse,
            gpu_available=self.config.gpu_enabled,
            npu_available=npu_available,
        )

        self.executor = Executor(
            dispatcher=self.dispatcher,
            pulse=self.pulse,
            monitor=self.monitor,
            cpu_workers=self.config.cpu_workers,
        )

        self._started = False

    def start(self):
        """Start the engine -- monitor and executor threads."""
        if self._started:
            return
        self.monitor.start()
        self.executor.start()
        self._started = True

    def stop(self):
        """Stop the engine cleanly."""
        if not self._started:
            return
        self.executor.stop()
        self.monitor.stop()
        self.personality.close()
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def run(self, operation: str, fn: Callable, *args, input_size: int = 0, **kwargs) -> Any:
        """Run a single operation through the engine.

        The dispatcher chooses the device, the executor runs it, and the
        personality records the result for future optimization.
        """
        if not self._started:
            self.start()

        work = WorkItem(
            operation=operation,
            fn=fn,
            args=args,
            kwargs=kwargs,
            input_size=input_size,
        )
        fragment = self.executor.execute(work)

        if fragment.error:
            raise RuntimeError(f"{operation} failed on {fragment.device.value}: {fragment.error}")

        return fragment.result

    def run_parallel(self, items: list[tuple[str, Callable, tuple]]) -> list[Any]:
        """Run multiple operations in parallel across devices.

        Args:
            items: List of (operation_name, function, args) tuples.
        """
        if not self._started:
            self.start()

        works = [
            WorkItem(operation=op, fn=fn, args=args)
            for op, fn, args in items
        ]
        fragments = self.executor.execute_parallel(works)
        results = []
        for f in fragments:
            if f.error:
                raise RuntimeError(f"{f.operation} failed on {f.device.value}: {f.error}")
            results.append(f.result)
        return results

    def benchmark(self, n: int = 10) -> dict:
        """Run a diagnostic benchmark across all available devices."""
        import numpy as np

        if not self._started:
            self.start()

        results = {"n": n, "devices": {}, "pulse": self.pulse.stats}
        size = 512

        # CPU matmul
        def cpu_matmul():
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            return a @ b

        cpu_times = []
        for _ in range(n):
            start = time.perf_counter()
            cpu_matmul()
            cpu_times.append((time.perf_counter() - start) * 1000)

        results["devices"]["cpu"] = {
            "matmul_512x512": {
                "avg_ms": round(sum(cpu_times) / len(cpu_times), 2),
                "min_ms": round(min(cpu_times), 2),
                "max_ms": round(max(cpu_times), 2),
            }
        }

        # GPU matmul via Kompute 0.9.0 (if available)
        try:
            import kp
            import struct
            mgr = kp.Manager()

            # Load pre-compiled SPIR-V shader
            spv_path = Path(__file__).parent / "shaders_matmul.spv"
            if not spv_path.exists():
                raise FileNotFoundError("shaders_matmul.spv not found -- compile with glslangValidator -V shaders_matmul.comp -o shaders_matmul.spv")
            spirv_bytes = spv_path.read_bytes()

            a_data = np.random.randn(size * size).astype(np.float32).tolist()
            b_data = np.random.randn(size * size).astype(np.float32).tolist()
            c_data = [0.0] * (size * size)

            t_a = mgr.tensor(a_data)
            t_b = mgr.tensor(b_data)
            t_c = mgr.tensor(c_data)
            params = [t_a, t_b, t_c]

            # Push constant: N as uint32 reinterpreted as float
            push_consts = [struct.unpack("f", struct.pack("I", size))[0]]

            gpu_times = []
            for i in range(n + 2):  # +2 warmup
                start = time.perf_counter()
                seq = mgr.sequence()
                seq.record(kp.OpSyncDevice(params))
                algo = mgr.algorithm(params, spirv_bytes, [size // 16, size // 16, 1], [], push_consts)
                seq.record(kp.OpAlgoDispatch(algo))
                seq.record(kp.OpSyncLocal([t_c]))
                seq.eval()
                elapsed = (time.perf_counter() - start) * 1000
                if i >= 2:  # skip warmup
                    gpu_times.append(elapsed)

            props = mgr.get_device_properties()
            results["devices"]["gpu"] = {
                "matmul_512x512": {
                    "avg_ms": round(sum(gpu_times) / len(gpu_times), 2),
                    "min_ms": round(min(gpu_times), 2),
                    "max_ms": round(max(gpu_times), 2),
                    "backend": "vulkan/kompute",
                    "device": props.get("device_name", "unknown"),
                }
            }
        except ImportError:
            results["devices"]["gpu"] = {"status": "kompute not available"}
        except FileNotFoundError as e:
            results["devices"]["gpu"] = {"status": str(e)}
        except Exception as e:
            results["devices"]["gpu"] = {"status": f"error: {e}"}

        # System snapshot
        snap = self.monitor.snapshot
        results["system"] = snap.to_dict()
        results["personality"] = self.personality.show()

        return results

    def status(self) -> dict:
        """Get current engine status."""
        snap = self.monitor.snapshot
        return {
            "running": self._started,
            "system": snap.to_dict(),
            "dispatcher": self.dispatcher.stats,
            "pulse": self.pulse.stats,
            "personality": self.personality.show(),
            "config": self.config.to_dict(),
        }
