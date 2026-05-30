"""
A/B benchmark: Neural scheduler vs HDC scheduler.
Runs the same workload through both and compares decision quality and speed.
"""
import time
import numpy as np
import sys
import os

# Make the repo root importable regardless of where it's checked out.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from engine.npu_scheduler import NpuScheduler  # noqa: E402
from engine.hdc_scheduler import HdcScheduler  # noqa: E402


def generate_workload(n_ops=100):
    """Generate a realistic mix of operations with varying system states."""
    ops = []
    gpu_temp = 45.0  # Starting temp

    for i in range(n_ops):
        op_types = ['matmul', 'attention', 'embed', 'normalize', 'conv', 'other']
        op_type = np.random.choice(op_types, p=[0.25, 0.2, 0.15, 0.15, 0.15, 0.1])

        # Simulate GPU heating up during compute-heavy ops
        if i > 0 and ops[-1]['op_type'] in ('matmul', 'attention', 'conv'):
            gpu_temp += np.random.uniform(0.5, 2.0)
        else:
            gpu_temp -= np.random.uniform(0.2, 1.0)
        gpu_temp = np.clip(gpu_temp, 40, 95)

        ops.append({
            'op_type': op_type,
            'op_size': np.random.randint(100, 1000000),
            'gpu_temp': gpu_temp,
            'gpu_util': np.random.uniform(20, 95),
            'cpu_load': np.random.uniform(10, 80),
            'npu_available': True,
        })
    return ops


def simulate_latency(op, device):
    """Simulate realistic latency for an op on a device."""
    base = {
        'cpu': {'matmul': 12.0, 'attention': 15.0, 'embed': 1.0, 'normalize': 0.5, 'conv': 8.0, 'other': 2.0},
        'gpu': {'matmul': 0.8, 'attention': 2.3, 'embed': 0.5, 'normalize': 0.3, 'conv': 1.5, 'other': 1.0},
        'npu': {'matmul': 3.0, 'attention': 5.0, 'embed': 0.3, 'normalize': 0.2, 'conv': 4.0, 'other': 1.5},
    }
    latency = base[device].get(op['op_type'], 2.0)

    # GPU penalty when hot
    if device == 'gpu' and op['gpu_temp'] > 80:
        latency *= 1.5 + (op['gpu_temp'] - 80) * 0.1

    # Size scaling
    latency *= (op['op_size'] / 100000) ** 0.3

    # Add noise
    latency *= np.random.uniform(0.8, 1.2)
    return latency


def benchmark():
    neural = NpuScheduler()
    hdc = HdcScheduler()

    # Load neural scheduler if saved
    neural_path = os.path.expanduser("~/.rag-race-router/scheduler.npz")
    if os.path.exists(neural_path):
        neural.load(neural_path)
        print("Loaded trained neural scheduler")

    workload = generate_workload(200)

    print("=" * 70)
    print("R.A.G-Race-Router: Neural vs HDC Scheduler Benchmark")
    print("=" * 70)
    print(f"Workload: {len(workload)} operations")
    print()

    # --- Phase 1: Train both on first 100 ops ---
    print("--- Training phase (100 ops) ---")

    neural_train_decisions = []
    hdc_train_decisions = []

    for op in workload[:100]:
        # Neural decision
        metrics_vec = np.array([
            op['gpu_temp'] / 100.0,
            op['gpu_util'] / 100.0,
            0.5,  # vram
            op['cpu_load'] / 100.0,
            1.0,  # npu available
            min(op['op_size'] / 1e6, 1.0)
        ], dtype=np.float32)

        neural_device, neural_probs = neural.forward(metrics_vec)
        neural_latency = simulate_latency(op, neural_device)
        neural.update(metrics_vec, neural_device, -neural_latency / 100.0)
        neural_train_decisions.append((neural_device, neural_latency))

        # HDC decision
        hdc_device, hdc_conf = hdc.dispatch(op)
        hdc_latency = simulate_latency(op, hdc_device)
        hdc.record(op, hdc_device, hdc_latency)
        hdc_train_decisions.append((hdc_device, hdc_latency))

    neural_avg_train = np.mean([d[1] for d in neural_train_decisions])
    hdc_avg_train = np.mean([d[1] for d in hdc_train_decisions])
    print(f"  Neural avg latency: {neural_avg_train:.2f}ms")
    print(f"  HDC avg latency:    {hdc_avg_train:.2f}ms")
    print()

    # --- Phase 2: Test on next 100 ops ---
    print("--- Test phase (100 ops) ---")

    neural_times = []
    hdc_times = []
    neural_latencies = []
    hdc_latencies = []

    for op in workload[100:]:
        # Neural: time the decision
        t0 = time.perf_counter_ns()
        metrics_vec = np.array([
            op['gpu_temp'] / 100.0, op['gpu_util'] / 100.0, 0.5,
            op['cpu_load'] / 100.0, 1.0, min(op['op_size'] / 1e6, 1.0)
        ], dtype=np.float32)
        neural_device, neural_probs = neural.forward(metrics_vec)
        neural_decision_ns = time.perf_counter_ns() - t0
        neural_times.append(neural_decision_ns)
        neural_latencies.append(simulate_latency(op, neural_device))

        # HDC: time the decision
        t0 = time.perf_counter_ns()
        hdc_device, hdc_conf = hdc.dispatch(op)
        hdc_decision_ns = time.perf_counter_ns() - t0
        hdc_times.append(hdc_decision_ns)
        hdc_latencies.append(simulate_latency(op, hdc_device))

    print(f"  {'Metric':<25} {'Neural':<15} {'HDC':<15} {'Winner':<10}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10}")

    neural_avg = np.mean(neural_latencies)
    hdc_avg = np.mean(hdc_latencies)
    winner = "HDC" if hdc_avg <= neural_avg else "Neural"
    print(f"  {'Avg op latency':<25} {neural_avg:<15.2f}ms {hdc_avg:<15.2f}ms {winner:<10}")

    neural_p99 = np.percentile(neural_latencies, 99)
    hdc_p99 = np.percentile(hdc_latencies, 99)
    winner = "HDC" if hdc_p99 <= neural_p99 else "Neural"
    print(f"  {'P99 op latency':<25} {neural_p99:<15.2f}ms {hdc_p99:<15.2f}ms {winner:<10}")

    neural_dec = np.mean(neural_times) / 1000  # ns to us
    hdc_dec = np.mean(hdc_times) / 1000
    winner = "HDC" if hdc_dec <= neural_dec else "Neural"
    print(f"  {'Decision speed':<25} {neural_dec:<15.1f}us {hdc_dec:<15.1f}us {winner:<10}")

    neural_total = np.sum(neural_latencies)
    hdc_total = np.sum(hdc_latencies)
    winner = "HDC" if hdc_total <= neural_total else "Neural"
    print(f"  {'Total workload':<25} {neural_total:<15.2f}ms {hdc_total:<15.2f}ms {winner:<10}")

    print()
    print("HDC Codebook:")
    print(hdc.show())
    print()

    # Device distribution comparison
    neural_devices = {}
    hdc_devices = {}
    for op in workload[100:]:
        metrics_vec = np.array([
            op['gpu_temp'] / 100.0, op['gpu_util'] / 100.0, 0.5,
            op['cpu_load'] / 100.0, 1.0, min(op['op_size'] / 1e6, 1.0)
        ], dtype=np.float32)
        nd, _ = neural.forward(metrics_vec)
        hd, _ = hdc.dispatch(op)
        neural_devices[nd] = neural_devices.get(nd, 0) + 1
        hdc_devices[hd] = hdc_devices.get(hd, 0) + 1

    print("Device distribution (test phase):")
    print(f"  {'Device':<8} {'Neural':<15} {'HDC':<15}")
    print(f"  {'-'*8} {'-'*15} {'-'*15}")
    for dev in ['cpu', 'gpu', 'npu']:
        nc = neural_devices.get(dev, 0)
        hc = hdc_devices.get(dev, 0)
        print(f"  {dev:<8} {nc:<15} {hc:<15}")

    # Save results
    results = {
        'neural_avg_latency': float(neural_avg),
        'hdc_avg_latency': float(hdc_avg),
        'neural_p99_latency': float(neural_p99),
        'hdc_p99_latency': float(hdc_p99),
        'neural_decision_us': float(neural_dec),
        'hdc_decision_us': float(hdc_dec),
        'neural_total_ms': float(neural_total),
        'hdc_total_ms': float(hdc_total),
        'codebook_entries': len(hdc.codebook),
        'workload_size': len(workload),
        'neural_device_dist': neural_devices,
        'hdc_device_dist': hdc_devices,
        'winner': 'HDC' if hdc_avg <= neural_avg else 'Neural',
    }

    import json
    with open('/tmp/hdc_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print("Results saved to /tmp/hdc_benchmark_results.json")
    print(f"\nOverall winner: {results['winner']}")
    return results


if __name__ == "__main__":
    benchmark()
