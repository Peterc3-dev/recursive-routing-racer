"""CLI for the R.A.G-Race-Router pulsed inference engine.

Usage:
    python -m engine --demo                    # Run demo workload once
    python -m engine --demo --runs 10          # Run 10 times (builds personality)
    python -m engine --demo --verbose          # Verbose dispatch logging
    python -m engine --personality show         # Show learned hardware profile
    python -m engine --status                  # Live system status
    python -m engine --benchmark               # Raw benchmark
"""

import argparse
import json
import os
import sys
import time

from . import EngineConfig, RagRaceRouter
from .dispatcher import Device


def main():
    parser = argparse.ArgumentParser(
        prog="rag-race-router",
        description="R.A.G-Race-Router -- Adaptive Tri-Processor Inference Runtime",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run three-processor dispatch demo workload",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run diagnostic benchmark across all devices",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of demo/benchmark iterations (default: 1)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose dispatch logging",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current engine and hardware status",
    )
    parser.add_argument(
        "--personality", choices=["show", "reset", "update"],
        help="Show, reset, or update routing personality",
    )
    parser.add_argument(
        "--config", choices=["show", "default"],
        help="Show current config or print default config",
    )
    parser.add_argument(
        "--stress", action="store_true",
        help="Run thermal stress test with adaptive rerouting",
    )
    parser.add_argument(
        "--duration", type=int, default=60,
        help="Stress test duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--gaming", action="store_true",
        help="Run gaming frame pipeline simulation",
    )
    parser.add_argument(
        "--frames", type=int, default=100,
        help="Number of gaming frames to simulate (default: 100)",
    )
    parser.add_argument(
        "--musicgen", action="store_true",
        help="Generate music through the engine (compare vs standalone)",
    )
    parser.add_argument(
        "--musicgen-tri", action="store_true",
        help="Run MusicGen through tri-processor analysis pipeline",
    )
    parser.add_argument("-p", "--prompt", type=str, default="deep bass dark atmospheric",
                        help="MusicGen prompt")
    parser.add_argument("--preset", type=str, help="MusicGen preset (electronic, ambient, etc.)")
    parser.add_argument("-d", type=float, default=10.0, help="MusicGen duration in seconds")
    parser.add_argument("-o", "--output", type=str, help="Output file path")
    parser.add_argument(
        "--analyze", type=str, metavar="MODEL.onnx",
        help="Analyze an ONNX model and show per-operator device routing",
    )
    parser.add_argument(
        "--train-scheduler", action="store_true",
        help="Train the neural dispatch scheduler via simulated workloads",
    )
    parser.add_argument(
        "--scheduler", choices=["show", "benchmark", "hdc"],
        help="Show scheduler policy or benchmark latency",
    )
    parser.add_argument(
        "--hdc-benchmark", action="store_true",
        help="Run Neural vs HDC scheduler A/B benchmark",
    )
    parser.add_argument("--config-file", type=str, help="Path to JSON config file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--temp-ceiling", type=float, help="GPU temperature ceiling")
    parser.add_argument("--burst-ms", type=float, help="GPU burst duration")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU execution")

    args = parser.parse_args()

    # Build config
    if args.config_file:
        config = EngineConfig.from_file(args.config_file)
    else:
        config = EngineConfig()

    if args.temp_ceiling:
        config.temp_ceiling = args.temp_ceiling
    if args.burst_ms:
        config.gpu_burst_ms = args.burst_ms
    if args.no_gpu:
        config.gpu_enabled = False

    # Handle config commands
    if args.config:
        if args.config == "show":
            _output(config.to_dict(), args.json)
        elif args.config == "default":
            _output(EngineConfig().to_dict(), args.json)
        return

    # Handle personality commands
    if args.personality:
        from .personality import Personality
        p = Personality()
        if args.personality == "show":
            if args.json:
                _output(p.show(), True)
            else:
                print(p.show_table())
        elif args.personality == "reset":
            import os
            from .personality import DB_PATH
            p.close()
            if DB_PATH.exists():
                os.remove(DB_PATH)
                print("Personality reset.")
            else:
                print("No personality data found.")
        elif args.personality == "update":
            p.update_rules(min_samples=3)
            print("Routing rules updated.")
            print(p.show_table())
        p.close()
        return

    # Engine operations
    engine = RagRaceRouter(config=config)

    if args.status:
        _run_status(engine, args.json)
        return

    if args.benchmark:
        print(f"Running benchmark ({args.runs} iterations)...")
        engine.start()
        time.sleep(0.5)
        results = engine.benchmark(n=args.runs)
        _output(results, args.json)
        engine.stop()
        return

    if args.demo:
        _run_demo(engine, args.runs, args.verbose, args.json)
        return

    if args.stress:
        from .stress_test import run_stress_test
        results = run_stress_test(engine, duration_s=args.duration, verbose=args.verbose)
        if args.json:
            _output(results, True)
        return

    if args.gaming:
        from .gaming_demo import run_gaming_benchmark
        print(f"R.A.G-Race-Router -- Gaming Simulation ({args.frames} frames)")
        print("-" * 55)
        print()
        results = run_gaming_benchmark(engine, n_frames=args.frames, verbose=args.verbose)

        if args.json:
            _output(results, True)
        else:
            e = results["engine"]
            c = results["cpu_baseline"]
            print("\nFrame pipeline: logic(CPU) -> geometry(GPU) -> shade(GPU) -> upscale(NPU) -> post(CPU)")
            print()
            print("CPU-only baseline:")
            print(f"  Avg frame: {c['avg_ms']:.1f}ms ({c['avg_fps']:.0f} FPS)")
            print(f"  1% low:    {c['p1_low_ms']:.1f}ms ({c['p1_low_fps']:.0f} FPS)")
            print()
            print("Engine (pulsed three-processor):")
            print(f"  Avg frame: {e['avg_ms']:.1f}ms ({e['avg_fps']:.0f} FPS)")
            print(f"  1% low:    {e['p1_low_ms']:.1f}ms ({e['p1_low_fps']:.0f} FPS)")
            print(f"  Peak GPU:  {results['peak_gpu_temp']:.0f}C")
            print(f"  Device map: {results['device_map']}")
        return

    if args.analyze:
        _run_analyze(args.analyze, args.json)
        return

    if args.train_scheduler:
        _run_train_scheduler(engine, args.runs or 50, args.verbose, args.json)
        return

    if args.scheduler:
        _run_scheduler_cmd(args.scheduler, args.json)
        return

    if getattr(args, 'musicgen_tri', False):
        from .musicgen_tri import run_tri_processor
        results = run_tri_processor(
            prompt=args.prompt,
            duration=args.d,
            preset=args.preset,
            output_dir=os.path.dirname(args.output) if args.output else "/tmp",
        )
        if args.json:
            _output(results, True)
        return

    if args.hdc_benchmark:
        from .hdc_benchmark import benchmark as hdc_bench
        results = hdc_bench()
        if args.json:
            _output(results, True)
        return

    if args.musicgen:
        from .musicgen_router import generate_standalone, generate_engine_routed
        print("R.A.G-Race-Router -- MusicGen Comparison")
        print("-" * 45)
        print()

        prompt = args.prompt
        duration = args.d
        preset = args.preset
        out_standalone = "/tmp/standalone.mp3"
        out_engine = args.output or "/tmp/engine.mp3"

        print(f"Prompt: {prompt}")
        if preset:
            print(f"Preset: {preset}")
        print(f"Duration: {duration}s")
        print()

        print("Generating standalone (baseline)...")
        standalone = generate_standalone(prompt, duration, preset, out_standalone)
        print(f"  Done: {standalone.get('elapsed_s', '?')}s -> {out_standalone}")

        print("Generating engine-routed...")
        routed = generate_engine_routed(prompt, duration, preset, out_engine)
        print(f"  Done: {routed.get('elapsed_s', '?')}s -> {out_engine}")

        print()
        print(f"{'Metric':<20} {'Standalone':<15} {'Engine':<15}")
        print("-" * 50)
        print(f"{'Time (s)':<20} {standalone.get('elapsed_s', 'N/A'):<15} {routed.get('elapsed_s', 'N/A'):<15}")
        print(f"{'Temp delta':<20} {'N/A':<15} {routed.get('temp_delta', 'N/A'):<15}")

        if args.json:
            _output({"standalone": standalone, "engine": routed}, True)
        return

    # No command specified
    parser.print_help()


def _run_status(engine: RagRaceRouter, as_json: bool):
    """Show live system status with all three processors."""

    engine.start()
    time.sleep(1.0)
    snap = engine.monitor.snapshot

    if as_json:
        _output(engine.status(), True)
        engine.stop()
        return

    gpu_ok = engine.config.gpu_enabled
    npu_ok = engine.dispatcher.npu_available
    cpu_ok = True

    print("R.A.G-Race-Router -- System Status")
    print("-" * 50)
    print(f"Processors: CPU {'ok' if cpu_ok else 'X'}  "
          f"GPU (Vulkan) {'ok' if gpu_ok else 'X'}  "
          f"NPU (XDNA) {'ok' if npu_ok else 'X'}")
    print()
    print(f"CPU:  {snap.cpu.freq_mhz:.0f} MHz | {snap.cpu.temp_c:.0f}C | "
          f"load {snap.cpu.load_avg[0]:.1f}")
    print(f"GPU:  {snap.gpu.temp_c:.0f}C | {snap.gpu.power_w:.1f}W | "
          f"VRAM {snap.gpu.vram_used_mb:.0f}/{snap.gpu.vram_total_mb:.0f} MB | "
          f"util {snap.gpu.util_pct:.0f}%")
    if npu_ok:
        print(f"NPU:  {snap.npu.status} | {snap.npu.driver} | "
              f"FLM {snap.npu.flm_version}")
    print(f"RAM:  {snap.mem_used_mb:.0f}/{snap.mem_total_mb:.0f} MB")
    print()

    pulse = engine.pulse.stats
    print(f"Pulse: duty {pulse['duty_cycle']:.1%} | "
          f"{pulse['burst_count']} bursts | "
          f"budget {engine.pulse.thermal_budget(snap.gpu.temp_c):.0%}")
    print(f"Personality: {engine.personality.show()['total_runs']} runs recorded")

    engine.stop()


def _run_demo(engine: RagRaceRouter, runs: int, verbose: bool, as_json: bool):
    """Run the three-processor dispatch demo."""
    from .ops import (
        build_demo_workload,
    )
    from .executor import WorkItem

    engine.start()
    time.sleep(1.5)  # Wait for monitor to collect GPU metrics via amdgpu_top

    snap = engine.monitor.snapshot
    gpu_ok = engine.config.gpu_enabled
    npu_ok = engine.dispatcher.npu_available

    if not as_json:
        print("R.A.G-Race-Router -- Three-Processor Pulsed Inference Demo")
        print("-" * 58)
        print()
        print(f"Processors: CPU ok  "
              f"GPU (Vulkan) {'ok' if gpu_ok else 'X'}  "
              f"NPU (XDNA) {'ok' if npu_ok else 'X'}")
        print(f"GPU Temp: {snap.gpu.temp_c:.0f}C | "
              f"VRAM: {snap.gpu.vram_used_mb:.0f}/{snap.gpu.vram_total_mb:.0f} MB | "
              f"Pulse: {'READY' if engine.pulse.should_fire_gpu(snap.gpu.temp_c) else 'COOLDOWN'}")
        print()

    all_results = []

    for run_idx in range(runs):
        if not as_json and runs > 1:
            print(f"--- Run {run_idx + 1}/{runs} ---")

        workload = build_demo_workload()

        # Execute pipeline with dependency tracking
        results = {}
        dispatch_log = []
        total_by_device = {"cpu": 0.0, "gpu": 0.0, "npu": 0.0}
        overlapped = 0

        for step in workload:
            name = step["name"]
            deps = step["deps"]

            # Resolve function and args based on device decision
            fn, args, device, reason = _resolve_op(
                engine, step, results, gpu_ok, npu_ok
            )

            # Execute directly on the chosen belt (bypass double-dispatch)
            work = WorkItem(
                operation=name,
                fn=fn,
                args=args if args else (),
                input_size=step["input_size"],
            )

            start_t = time.perf_counter()
            fragment = engine.executor.execute(work)
            elapsed = (time.perf_counter() - start_t) * 1000

            results[name] = fragment.result
            actual_device = fragment.device
            total_by_device[actual_device.value] += elapsed

            # Check if this could overlap with previous
            if not deps:
                overlapped += 1

            tag = ""
            if actual_device == Device.GPU:
                tag = " <- pulsed burst"
            elif actual_device == Device.NPU:
                tag = " <- npu"

            dispatch_log.append({
                "op": name,
                "device": actual_device.value,
                "ms": round(elapsed, 2),
                "tag": tag,
                "reason": reason,
            })

            if verbose and not as_json:
                print(f"  {name:<14} -> {actual_device.value:<6} ({elapsed:.2f}ms) "
                      f"[{reason}]{tag}")

        total_ms = sum(d["ms"] for d in dispatch_log)
        gpu_ms = total_by_device["gpu"]
        npu_ms = total_by_device["npu"]
        cpu_ms = total_by_device["cpu"]
        parallel_eff = (overlapped / max(len(workload), 1)) * 100

        snap_after = engine.monitor.snapshot
        temp_delta = snap_after.gpu.temp_c - snap.gpu.temp_c

        run_result = {
            "run": run_idx + 1,
            "ops": dispatch_log,
            "total_ms": round(total_ms, 2),
            "gpu_ms": round(gpu_ms, 2),
            "npu_ms": round(npu_ms, 2),
            "cpu_ms": round(cpu_ms, 2),
            "parallel_ops": overlapped,
            "temp_before": snap.gpu.temp_c,
            "temp_after": snap_after.gpu.temp_c,
        }
        all_results.append(run_result)

        if not as_json and not verbose:
            print(f"Dispatching {len(workload)} operations:")
            for d in dispatch_log:
                print(f"  {d['op']:<14} -> {d['device']:<6} ({d['ms']:.2f}ms){d['tag']}")

        if not as_json:
            print()
            print(f"Pipeline: {total_ms:.2f}ms total "
                  f"({gpu_ms:.2f}ms GPU, {npu_ms:.2f}ms NPU, {cpu_ms:.2f}ms CPU)")
            print(f"Parallel efficiency: {parallel_eff:.0f}% ({overlapped} ops overlapped)")
            print(f"GPU temp after: {snap_after.gpu.temp_c:.0f}C "
                  f"({'+' if temp_delta >= 0 else ''}{temp_delta:.0f}C)")
            print()

        # Record to personality
        engine.personality.update_rules(min_samples=3)

        snap = snap_after

    # Final personality status
    p_data = engine.personality.show()
    if not as_json:
        rules_status = "adaptive routing active" if engine.dispatcher._rules_encoded else \
            f"Need {max(0, engine.dispatcher.LEARNING_THRESHOLD - runs)} more for adaptive routing"
        print(f"Personality: {p_data['total_runs']} runs recorded. {rules_status}.")
        if runs >= engine.dispatcher.LEARNING_THRESHOLD and p_data["routing_rules"]:
            print()
            print(engine.personality.show_table())

    if as_json:
        _output({"runs": all_results, "personality": p_data}, True)

    engine.stop()


def _resolve_op(engine, step, results, gpu_ok, npu_ok):
    """Resolve function, args, and device for a workload step. Returns (fn, args, device, reason)."""
    from .ops import (
        cpu_embed, cpu_attention,
        cpu_normalize, cpu_project, cpu_decode,
        gpu_attention, gpu_project,
        npu_embed, npu_normalize,
    )
    import numpy as np

    name = step["name"]

    # Determine device via dispatcher (single call)
    decision = engine.dispatcher.dispatch(name, step["input_size"])
    device = decision.device
    reason = decision.reason

    # Resolve function and args based on device + dependencies
    if name == "tokenize":
        fn, args = step["cpu"]
        return fn, args, Device.CPU, reason

    if name == "embed":
        tokens = results.get("tokenize")
        if device == Device.NPU and npu_ok:
            return npu_embed, (tokens, 512), Device.NPU, reason
        return cpu_embed, (tokens, 512), Device.CPU, reason

    if name == "matmul":
        if device == Device.GPU and gpu_ok:
            return step["gpu"][0], step["gpu"][1], Device.GPU, reason
        return step["cpu"][0], step["cpu"][1], Device.CPU, reason

    if name == "attention":
        embed_result = results.get("embed")
        if embed_result is not None and len(embed_result.shape) == 2:
            q = embed_result[:min(16, embed_result.shape[0])]
            k, v = q.copy(), q.copy()
        else:
            q = k = v = np.random.randn(16, 512).astype(np.float32)
        if device == Device.GPU and gpu_ok:
            return gpu_attention, (q, k, v), Device.GPU, reason
        return cpu_attention, (q, k, v), Device.CPU, reason

    if name == "normalize":
        attn_result = results.get("attention")
        if attn_result is None:
            attn_result = np.random.randn(16, 512).astype(np.float32)
        if device == Device.NPU and npu_ok:
            return npu_normalize, (attn_result,), Device.NPU, reason
        return cpu_normalize, (attn_result,), Device.CPU, reason

    if name == "project":
        norm_result = results.get("normalize")
        if norm_result is None:
            norm_result = np.random.randn(16, 512).astype(np.float32)
        weight = np.random.randn(norm_result.shape[-1], 128).astype(np.float32)
        if device == Device.GPU and gpu_ok:
            return gpu_project, (norm_result, weight), Device.GPU, reason
        return cpu_project, (norm_result, weight), Device.CPU, reason

    if name == "decode":
        proj_result = results.get("project")
        if proj_result is None:
            proj_result = np.random.randn(16, 128).astype(np.float32)
        return cpu_decode, (proj_result,), Device.CPU, reason

    # Fallback
    if step["cpu"]:
        fn, args = step["cpu"]
        return fn, args, Device.CPU, reason
    raise ValueError(f"No implementation for {name}")


def _output(data: dict, as_json: bool):
    if as_json:
        print(json.dumps(data, indent=2, default=str))
    else:
        _pretty_print(data)


def _pretty_print(data: dict, indent: int = 0):
    prefix = "  " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _pretty_print(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    _pretty_print(item, indent + 1)
                    print(f"{prefix}  ---")
                else:
                    print(f"{prefix}  - {item}")
        else:
            print(f"{prefix}{key}: {value}")


def _run_analyze(model_path: str, as_json: bool):
    """Analyze an ONNX model and display per-operator routing."""
    from .onnx_dispatcher import OnnxDispatcher

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        sys.exit(1)

    d = OnnxDispatcher(model_path)

    if as_json:
        _output({
            "model": os.path.basename(model_path),
            "routing": d.get_routing(),
            "rules": d.export_rules(),
            "op_summary": d.op_type_summary(),
            "summary": d.summary(),
        }, True)
    else:
        print(f"ONNX Model Analysis: {os.path.basename(model_path)}")
        print("=" * 55)
        print(d.summary())
        print()

        # Show per-op-type breakdown
        ops = d.op_type_summary()
        print(f"{'Op Type':<25} {'Device':<8} {'Count':<8} {'Avg Size':<12}")
        print("-" * 55)
        for op, info in sorted(ops.items(), key=lambda x: -x[1]["count"]):
            avg_size = int(sum(info["sizes"]) / len(info["sizes"])) if info["sizes"] else 0
            size_str = f"{avg_size:,}" if avg_size else "-"
            print(f"{op:<25} {info['device']:<8} {info['count']:<8} {size_str:<12}")
        print()

        rules = d.export_rules()
        print(f"Routing rules: {len(rules)} op types mapped")
        print(f"  GPU ops: {sum(1 for v in rules.values() if v == 'gpu')}")
        print(f"  NPU ops: {sum(1 for v in rules.values() if v == 'npu')}")
        print(f"  CPU ops: {sum(1 for v in rules.values() if v == 'cpu')}")


def _run_train_scheduler(engine: RagRaceRouter, runs: int, verbose: bool, as_json: bool):
    """Train the neural scheduler by running simulated workloads."""
    from .npu_scheduler import NpuScheduler
    from .ops import build_demo_workload
    import numpy as np

    scheduler = NpuScheduler()
    weights_path = os.path.expanduser("~/.rag-race-router/scheduler.npz")
    scheduler.load(weights_path)

    engine.start()
    time.sleep(1.0)

    print(f"Training neural scheduler ({runs} runs)...")
    print(f"  Weights: {weights_path}")
    print(f"  Params: {scheduler.param_count}")
    print()

    for run_idx in range(runs):
        snap = engine.monitor.snapshot
        workload = build_demo_workload()

        for step in workload:
            op = step["name"]
            size = step["input_size"]

            metrics = np.array([
                min(snap.gpu.temp_c / 100.0, 1.0),
                min(snap.gpu.util_pct / 100.0, 1.0),
                min(snap.gpu.vram_used_mb / max(snap.gpu.vram_total_mb, 1), 1.0),
                min(snap.cpu.load_avg[0] / 100.0, 1.0) if snap.cpu.load_avg else 0.5,
                1.0 if engine.dispatcher.npu_available else 0.0,
                min(size / 1e6, 1.0),
            ], dtype=np.float32)

            device, probs = scheduler.forward(metrics)

            # Simulate latency based on device + op characteristics
            # Benchmarked: GPU 1084 GFLOPS, CPU 27 GFLOPS, NPU 50 TOPS INT8
            size_norm = min(size / 1e6, 1.0)
            is_compute = op in ("matmul", "attention", "project", "conv2d", "gemm")
            is_small = size < 256 * 256
            is_activation = op in ("normalize", "softmax", "relu", "embed", "layernorm")

            if device == "gpu":
                if is_compute:
                    latency = 0.1 + size_norm * 0.5  # GPU excels at large compute
                else:
                    latency = 0.8 + size_norm * 0.3  # GPU overhead for non-compute
            elif device == "npu":
                if is_activation or is_small:
                    latency = 0.05 + size_norm * 0.2  # NPU best for small/activation ops
                elif is_compute:
                    latency = 0.3 + size_norm * 2.0  # NPU slower for large compute
                else:
                    latency = 0.2 + size_norm * 0.4
            else:  # cpu
                if is_compute:
                    latency = 2.0 + size_norm * 5.0  # CPU very slow for large compute
                elif is_activation:
                    latency = 0.5 + size_norm * 0.3  # CPU OK for activations
                else:
                    latency = 0.1 + size_norm * 0.1  # CPU fastest for control flow

            # Thermal penalty for GPU
            temp_norm = snap.gpu.temp_c / 100.0
            if device == "gpu" and temp_norm > 0.85:
                latency *= 1.0 + (temp_norm - 0.85) * 10.0
            # NPU unavailable penalty
            if device == "npu" and not engine.dispatcher.npu_available:
                latency *= 20.0

            reward = -latency
            scheduler.update(metrics, device, reward)

            if verbose:
                print(f"  [{run_idx+1}/{runs}] {op:<14} -> {device} "
                      f"(lat={latency:.1f}ms, reward={reward:.3f})")

        if (run_idx + 1) % 10 == 0 or run_idx == 0:
            print(f"  Run {run_idx+1}/{runs}: avg_reward={scheduler.running_reward:.4f}")

    scheduler.save(weights_path)
    engine.stop()

    print()
    print(f"Training complete. {scheduler.total_updates} updates.")
    print(f"Weights saved to: {weights_path}")
    print()
    print(scheduler.show_policy())

    if as_json:
        _output({
            "runs": runs,
            "updates": scheduler.total_updates,
            "running_reward": scheduler.running_reward,
            "weights_path": weights_path,
            "latency": scheduler.benchmark_latency(),
        }, True)


def _run_scheduler_cmd(cmd: str, as_json: bool):
    """Show scheduler status or run benchmark."""
    from .npu_scheduler import NpuScheduler

    scheduler = NpuScheduler()
    weights_path = os.path.expanduser("~/.rag-race-router/scheduler.npz")

    if scheduler.load(weights_path):
        print(f"Loaded scheduler from: {weights_path}")
    else:
        print(f"No saved scheduler found at: {weights_path}")
        print("Run --train-scheduler first.")
        return

    if cmd == "show":
        npu_status = scheduler.deploy_to_npu()
        print()
        print("NPU Scheduler Status:")
        print(f"  Weights: {weights_path}")
        print(f"  Training runs: {scheduler.total_updates}")
        print(f"  Running on: {npu_status}")
        print()
        print(scheduler.show_policy())

        if as_json:
            _output({
                "weights_path": weights_path,
                "total_updates": scheduler.total_updates,
                "running_reward": scheduler.running_reward,
                "param_count": scheduler.param_count,
                "npu_status": npu_status,
            }, True)

    elif cmd == "hdc":
        from .hdc_scheduler import HdcScheduler
        hdc = HdcScheduler()
        print()
        print(hdc.show())
        return

    elif cmd == "benchmark":
        results = scheduler.benchmark_latency()
        print()
        print(f"Scheduler Latency Benchmark ({results['n']} iterations):")
        print(f"  Avg: {results['avg_us']:.2f} us")
        print(f"  Min: {results['min_us']:.2f} us")
        print(f"  Max: {results['max_us']:.2f} us")
        print(f"  P99: {results['p99_us']:.2f} us")

        if as_json:
            _output(results, True)


if __name__ == "__main__":
    main()
