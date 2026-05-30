"""Thermal stress test — push GPU until throttle, watch dispatch adapt.

Runs compute-heavy ops at maximum rate, monitors temperature, and
demonstrates adaptive rerouting as GPU heats up.
"""

import sys
import time
import numpy as np
from dataclasses import dataclass


@dataclass
class StressEntry:
    t: float
    gpu_temp: float
    gpu_util: float
    device: str
    op: str
    ms: float
    rerouted: bool = False
    reason: str = ""


def run_stress_test(engine, duration_s: int = 120, verbose: bool = False) -> dict:
    """Run thermal stress test with live visualization."""
    from .executor import WorkItem
    from .ops import cpu_matmul, cpu_normalize

    engine.start()
    time.sleep(1.0)

    snap = engine.monitor.snapshot
    initial_temp = snap.gpu.temp_c

    log = []
    temp_curve = []
    reroute_count = 0
    total_ops = 0
    start_time = time.time()

    # Progress bar width
    BAR_W = 24

    print(f"R.A.G-Race-Router -- Thermal Stress Test ({duration_s}s)")
    print("-" * 55)
    print()

    try:
        while time.time() - start_time < duration_s:
            elapsed = time.time() - start_time
            snap = engine.monitor.snapshot
            temp = snap.gpu.temp_c
            temp_curve.append(round(temp, 1))

            # Determine stress level based on temperature
            if temp < 65:
                zone = "COOL"
                ops_to_run = ["matmul", "attention", "matmul"]  # All GPU
            elif temp < 75:
                zone = "WARM"
                ops_to_run = ["matmul", "normalize", "matmul"]  # Mix GPU+NPU
            elif temp < 85:
                zone = "HOT"
                ops_to_run = ["normalize", "matmul", "normalize"]  # More NPU
            else:
                zone = "CRITICAL"
                ops_to_run = ["normalize", "normalize", "normalize"]  # All NPU/CPU

            for op_name in ops_to_run:
                size = 512
                a = np.random.randn(size, size).astype(np.float32)
                b = np.random.randn(size, size).astype(np.float32)

                if op_name == "matmul":
                    fn = cpu_matmul
                    args = (a, b)
                else:
                    fn = cpu_normalize
                    args = (a,)

                work = WorkItem(operation=op_name, fn=fn, args=args,
                                input_size=size * size)

                t0 = time.perf_counter()
                frag = engine.executor.execute(work)
                op_ms = (time.perf_counter() - t0) * 1000

                device_used = frag.device.value
                was_rerouted = (op_name == "matmul" and device_used != "gpu") or \
                               (op_name == "normalize" and device_used not in ("npu", "cpu"))

                if was_rerouted:
                    reroute_count += 1

                entry = StressEntry(
                    t=elapsed, gpu_temp=temp, gpu_util=snap.gpu.util_pct,
                    device=device_used, op=op_name, ms=round(op_ms, 2),
                    rerouted=was_rerouted,
                    reason=f"thermal ({zone})" if was_rerouted else "",
                )
                log.append(entry)
                total_ops += 1

            # Live display
            temp_bar = int(min(temp / 100, 1.0) * BAR_W)
            temp_fill = "\u2588" * temp_bar + "\u2591" * (BAR_W - temp_bar)

            sys.stdout.write(
                f"\r  [{int(elapsed):3d}s/{duration_s}s] "
                f"GPU {temp_fill} {temp:.0f}C [{zone:8s}] "
                f"reroutes: {reroute_count}"
            )
            sys.stdout.flush()

            # Brief pause to not overwhelm
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    print()
    print()

    # Summary
    gpu_ops = sum(1 for e in log if e.device == "gpu")
    npu_ops = sum(1 for e in log if e.device == "npu")
    cpu_ops = sum(1 for e in log if e.device == "cpu")
    peak_temp = max(e.gpu_temp for e in log) if log else 0

    # Sample temp curve (every 5 entries)
    curve_sample = temp_curve[::max(1, len(temp_curve) // 20)]

    result = {
        "duration_s": round(time.time() - start_time, 1),
        "total_ops": total_ops,
        "reroutes": reroute_count,
        "initial_temp": initial_temp,
        "peak_temp": peak_temp,
        "final_temp": temp_curve[-1] if temp_curve else 0,
        "ops_by_device": {"gpu": gpu_ops, "npu": npu_ops, "cpu": cpu_ops},
        "temp_curve_sample": curve_sample,
    }

    print(f"Total ops: {total_ops} | Reroutes: {reroute_count}")
    print(f"Ops by device: GPU={gpu_ops}, NPU={npu_ops}, CPU={cpu_ops}")
    print(f"Temp: {initial_temp:.0f}C -> {peak_temp:.0f}C peak -> {result['final_temp']:.0f}C final")
    print(f"Curve: {' '.join(f'{t:.0f}' for t in curve_sample)}")

    engine.stop()
    return result
