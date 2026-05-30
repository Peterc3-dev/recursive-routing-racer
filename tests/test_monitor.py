"""Unit tests for the pure dataclass / serialization logic in engine/monitor.py.

The hardware-reading methods (_read_gpu, _read_npu, ...) shell out to
amdgpu_top / sysfs / journalctl and are intentionally NOT exercised here —
only the SystemSnapshot.to_dict() serialization is tested.
"""

from conftest import load_module

monitor = load_module("monitor")
GpuMetrics = monitor.GpuMetrics
CpuMetrics = monitor.CpuMetrics
NpuMetrics = monitor.NpuMetrics
SystemSnapshot = monitor.SystemSnapshot


def test_default_snapshot_to_dict_shape():
    d = SystemSnapshot().to_dict()
    assert set(d.keys()) == {
        "timestamp", "gpu", "cpu", "npu", "mem_used_mb", "mem_total_mb"
    }
    assert set(d["gpu"].keys()) == {
        "temp_c", "util_pct", "vram_used_mb", "vram_total_mb",
        "power_w", "clock_mhz", "throttling",
    }
    assert set(d["cpu"].keys()) == {
        "util_pct", "freq_mhz", "temp_c", "load_avg",
    }


def test_to_dict_preserves_values():
    snap = SystemSnapshot(
        timestamp=123.0,
        gpu=GpuMetrics(temp_c=65.5, util_pct=42.0, throttling=True, clock_mhz=2400),
        cpu=CpuMetrics(util_pct=10.0, load_avg=(1.0, 2.0, 3.0)),
        npu=NpuMetrics(present=True, status="available"),
        mem_used_mb=8000.0,
        mem_total_mb=16000.0,
    )
    d = snap.to_dict()
    assert d["timestamp"] == 123.0
    assert d["gpu"]["temp_c"] == 65.5
    assert d["gpu"]["throttling"] is True
    assert d["gpu"]["clock_mhz"] == 2400
    assert d["cpu"]["load_avg"] == [1.0, 2.0, 3.0]  # tuple -> list
    assert d["npu"]["present"] is True
    assert d["npu"]["status"] == "available"
    assert d["mem_used_mb"] == 8000.0


def test_load_avg_serializes_to_list():
    snap = SystemSnapshot(cpu=CpuMetrics(load_avg=(0.5, 0.6, 0.7)))
    d = snap.to_dict()
    assert isinstance(d["cpu"]["load_avg"], list)


def test_to_dict_is_json_serializable():
    import json
    snap = SystemSnapshot(npu=NpuMetrics(present=True, device_path="/dev/accel/accel0"))
    # Should not raise.
    text = json.dumps(snap.to_dict())
    assert "/dev/accel/accel0" in text
