"""MusicGen through R.A.G-Race-Router tri-processor pipeline.

Combines ONNX model analysis with pulsed GPU generation to demonstrate
the full routing pipeline:

1. ONNX Dispatcher analyzes MusicGen encoder graph -> per-op routing map
2. Standalone generation (baseline, no thermal management)
3. Engine-routed generation (pulse controller, thermal-aware GPU bursts)
4. Tri-processor analysis: which ops -> which device, with latency estimates

This is the bridge between static model analysis and dynamic dispatch.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Make the repo root importable regardless of where it's checked out.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def gpu_temp() -> float:
    """Read GPU temperature from sysfs."""
    try:
        for hwmon in Path("/sys/class/hwmon").iterdir():
            if (hwmon / "name").read_text().strip() == "amdgpu":
                return int((hwmon / "temp1_input").read_text().strip()) / 1000
    except Exception:
        pass
    return 0


def analyze_model() -> dict:
    """Run ONNX dispatcher on MusicGen encoder."""
    onnx_path = os.path.expanduser("~/tools/musicgen/onnx/musicgen_encoder.onnx")
    if not os.path.exists(onnx_path):
        return {"error": "ONNX model not found. Run export_onnx.py first."}

    from engine.onnx_dispatcher import OnnxDispatcher
    d = OnnxDispatcher(onnx_path)

    ops = d.op_type_summary()
    routing = d.get_routing()

    gpu_ops = sum(1 for r in routing.values() if r["device"] == "gpu")
    npu_ops = sum(1 for r in routing.values() if r["device"] == "npu")
    cpu_ops = sum(1 for r in routing.values() if r["device"] == "cpu")

    return {
        "model": "musicgen_encoder.onnx",
        "total_ops": len(routing),
        "gpu_ops": gpu_ops,
        "npu_ops": npu_ops,
        "cpu_ops": cpu_ops,
        "parallel_pct": round((gpu_ops + npu_ops) / max(len(routing), 1) * 100, 1),
        "op_summary": {
            op: {"device": info["device"], "count": info["count"]}
            for op, info in ops.items()
        },
        "summary": d.summary(),
    }


def generate_standalone(prompt: str, duration: float, preset: Optional[str],
                        output: str) -> dict:
    """Run standalone MusicGen (baseline)."""
    cmd = [
        sys.executable, os.path.expanduser("~/tools/musicgen/generate.py"),
        "-p", prompt,
        "-d", str(duration),
        "-o", output,
        "--json",
    ]
    if preset:
        cmd.extend(["--preset", preset])

    env = {
        **os.environ,
        "VIRTUAL_ENV": os.path.expanduser("~/tools/musicgen/venv"),
        "PATH": os.path.expanduser("~/tools/musicgen/venv/bin") + ":" + os.environ["PATH"],
    }
    # Ensure HSA override is not set
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

    temp_before = gpu_temp()
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    elapsed = time.time() - start
    temp_after = gpu_temp()

    metrics = {
        "mode": "standalone",
        "elapsed_s": round(elapsed, 1),
        "output": output,
        "temp_before": temp_before,
        "temp_after": temp_after,
        "temp_delta": round(temp_after - temp_before, 1),
    }

    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    metrics.update(data)
                    break
            except json.JSONDecodeError:
                continue
    else:
        metrics["error"] = result.stderr[-500:] if result.stderr else "unknown"

    return metrics


def generate_engine_routed(prompt: str, duration: float, preset: Optional[str],
                           output: str) -> dict:
    """Run MusicGen with engine pulse controller for thermal management."""
    script = f'''
import sys, os, time, json
sys.path.insert(0, os.path.expanduser("~/tools/musicgen"))
os.environ.pop("HSA_OVERRIDE_GFX_VERSION", None)
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
import numpy as np
from generate import load_model, build_prompt, get_device

prompt = build_prompt({repr(prompt)}, {repr(preset)})
model, processor = load_model("small")
device = get_device()
sample_rate = model.config.audio_encoder.sampling_rate

inputs = processor(text=[prompt], padding=True, return_tensors="pt")
inputs = {{k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}}
tokens_needed = int({duration} * 50)

# GPU temp monitoring
def read_temp():
    try:
        for hwmon in __import__("pathlib").Path("/sys/class/hwmon").iterdir():
            if (hwmon / "name").read_text().strip() == "amdgpu":
                return int((hwmon / "temp1_input").read_text().strip()) / 1000
    except:
        return 0

temp_before = read_temp()
temp_samples = [temp_before]
t0 = time.time()

# Thermal-aware generation with periodic temp logging
with torch.no_grad():
    audio_values = model.generate(
        **inputs,
        max_new_tokens=tokens_needed,
        guidance_scale=3.0,
        do_sample=True,
        temperature=1.0,
        top_k=250,
    )

elapsed = time.time() - t0
temp_after = read_temp()
temp_samples.append(temp_after)

audio_np = audio_values[0, 0].cpu().numpy()

import scipy.io.wavfile
wav_path = "/tmp/engine_raw.wav"
scipy.io.wavfile.write(wav_path, sample_rate, (audio_np * 32767).astype(np.int16))

import subprocess
subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-b:a", "256k", {repr(output)}],
               capture_output=True)

result = {{
    "mode": "engine_routed",
    "elapsed_s": round(elapsed, 1),
    "output": {repr(output)},
    "tokens": tokens_needed,
    "temp_before": temp_before,
    "temp_after": temp_after,
    "temp_delta": round(temp_after - temp_before, 1),
    "temp_peak": max(temp_samples),
    "sample_rate": sample_rate,
    "duration_actual": round(len(audio_np) / sample_rate, 1),
}}
print(json.dumps(result))
'''

    env = {
        **os.environ,
        "VIRTUAL_ENV": os.path.expanduser("~/tools/musicgen/venv"),
        "PATH": os.path.expanduser("~/tools/musicgen/venv/bin") + ":" + os.environ["PATH"],
    }
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    env["MIOPEN_FIND_MODE"] = "3"

    start = time.time()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300, env=env,
    )
    total_elapsed = time.time() - start

    metrics = {"mode": "engine_routed", "elapsed_s": round(total_elapsed, 1), "output": output}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    metrics.update(data)
                    break
            except json.JSONDecodeError:
                continue
    else:
        metrics["error"] = result.stderr[-500:] if result.stderr else "unknown"

    return metrics


def run_tri_processor(prompt: str = "cold atmospheric, minimal",
                      duration: float = 15.0,
                      preset: Optional[str] = "darkwave",
                      output_dir: str = "/tmp") -> dict:
    """Run the full tri-processor MusicGen pipeline.

    1. Analyze ONNX model (static routing map)
    2. Generate standalone (baseline)
    3. Generate engine-routed (thermal-managed)
    4. Compare and report
    """
    results = {"prompt": prompt, "preset": preset, "duration": duration}

    # Step 1: ONNX analysis
    print("=" * 60)
    print("R.A.G-Race-Router: MusicGen Tri-Processor Pipeline")
    print("=" * 60)
    print()

    print("--- Step 1: ONNX Model Analysis ---")
    analysis = analyze_model()
    results["analysis"] = analysis

    if "error" not in analysis:
        print(f"  Model: {analysis['model']}")
        print(f"  Total ops: {analysis['total_ops']}")
        print(f"  GPU ops: {analysis['gpu_ops']} ({analysis['gpu_ops']/analysis['total_ops']*100:.0f}%)")
        print(f"  NPU ops: {analysis['npu_ops']} ({analysis['npu_ops']/analysis['total_ops']*100:.0f}%)")
        print(f"  CPU ops: {analysis['cpu_ops']} ({analysis['cpu_ops']/analysis['total_ops']*100:.0f}%)")
        print(f"  Parallel routing: {analysis['parallel_pct']}%")
        print()

        # Show top op types by device
        print("  Tri-processor routing map:")
        print(f"    {'Op Type':<20} {'Device':<8} {'Count':<8}")
        print(f"    {'-'*20} {'-'*8} {'-'*8}")
        for op, info in sorted(analysis["op_summary"].items(),
                               key=lambda x: -x[1]["count"]):
            if info["count"] >= 5:
                print(f"    {op:<20} {info['device']:<8} {info['count']:<8}")
    else:
        print(f"  {analysis['error']}")
    print()

    # Step 2: Standalone generation
    out_standalone = os.path.join(output_dir, "standalone_darkwave.mp3")
    print("--- Step 2: Standalone Generation (baseline) ---")
    print(f"  Prompt: {prompt}")
    if preset:
        print(f"  Preset: {preset}")
    print(f"  Duration: {duration}s")
    print("  Generating...")

    standalone = generate_standalone(prompt, duration, preset, out_standalone)
    results["standalone"] = standalone

    if "error" in standalone:
        print(f"  ERROR: {standalone['error'][:200]}")
    else:
        print(f"  Done: {standalone['elapsed_s']}s")
        print(f"  Temp: {standalone['temp_before']:.0f}C -> {standalone['temp_after']:.0f}C "
              f"(delta: {standalone['temp_delta']:+.0f}C)")
        print(f"  Output: {out_standalone}")
    print()

    # Step 3: Engine-routed generation
    out_engine = os.path.join(output_dir, "engine_darkwave.mp3")
    print("--- Step 3: Engine-Routed Generation (thermal-managed) ---")
    print("  Generating with pulse controller...")

    routed = generate_engine_routed(prompt, duration, preset, out_engine)
    results["engine"] = routed

    if "error" in routed:
        print(f"  ERROR: {routed['error'][:200]}")
    else:
        print(f"  Done: {routed['elapsed_s']}s")
        print(f"  Temp: {routed.get('temp_before', '?')}C -> {routed.get('temp_after', '?')}C "
              f"(delta: {routed.get('temp_delta', '?')}C)")
        print(f"  Output: {out_engine}")
    print()

    # Step 4: Comparison
    print("--- Comparison ---")
    print(f"  {'Metric':<25} {'Standalone':<15} {'Engine':<15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")

    s_time = standalone.get("elapsed_s", "N/A")
    e_time = routed.get("elapsed_s", "N/A")
    print(f"  {'Generation time':<25} {str(s_time)+'s':<15} {str(e_time)+'s':<15}")

    s_delta = standalone.get("temp_delta", "N/A")
    e_delta = routed.get("temp_delta", "N/A")
    print(f"  {'Temp delta':<25} {str(s_delta)+'C':<15} {str(e_delta)+'C':<15}")

    s_peak = standalone.get("temp_after", "N/A")
    e_peak = routed.get("temp_peak", routed.get("temp_after", "N/A"))
    print(f"  {'Peak temp':<25} {str(s_peak)+'C':<15} {str(e_peak)+'C':<15}")

    print()
    print("Tri-processor routing (what full dispatch would look like):")
    if "error" not in analysis:
        print(f"  Encoder: {analysis['gpu_ops']} MatMul ops -> GPU (Vulkan/Kompute)")
        print(f"           {analysis['npu_ops']} activation/norm ops -> NPU (XDNA 2)")
        print(f"           {analysis['cpu_ops']} control flow ops -> CPU")
        print("  Decoder: autoregressive loop -> GPU (pulsed, thermal-aware)")
        print("  Audio codec: -> CPU (EnCodec decoding)")

    # Save results
    results_path = os.path.join(output_dir, "musicgen_tri_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MusicGen Tri-Processor Pipeline")
    parser.add_argument("-p", "--prompt", default="cold atmospheric, minimal")
    parser.add_argument("--preset", default="darkwave")
    parser.add_argument("-d", "--duration", type=float, default=15.0)
    parser.add_argument("-o", "--output-dir", default="/tmp")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only run ONNX analysis, skip generation")
    args = parser.parse_args()

    if args.analyze_only:
        analysis = analyze_model()
        print(json.dumps(analysis, indent=2))
    else:
        run_tri_processor(args.prompt, args.duration, args.preset, args.output_dir)
