"""Unit tests for the pure thermal-pulse logic in engine/pulse.py.

No GPU is involved — these exercise the temperature → budget → burst/cooldown
math only.
"""

import math

from conftest import load_module

pulse = load_module("pulse")
PulseConfig = pulse.PulseConfig
PulseController = pulse.PulseController


def test_thermal_budget_full_at_or_below_floor():
    pc = PulseController(PulseConfig(temp_floor=60.0, temp_ceiling=80.0))
    assert pc.thermal_budget(60.0) == 1.0
    assert pc.thermal_budget(50.0) == 1.0


def test_thermal_budget_zero_at_or_above_ceiling():
    pc = PulseController(PulseConfig(temp_floor=60.0, temp_ceiling=80.0))
    assert pc.thermal_budget(80.0) == 0.0
    assert pc.thermal_budget(95.0) == 0.0


def test_thermal_budget_linear_midpoint():
    pc = PulseController(PulseConfig(temp_floor=60.0, temp_ceiling=80.0))
    # Halfway between floor and ceiling -> half budget.
    assert math.isclose(pc.thermal_budget(70.0), 0.5, rel_tol=1e-9)


def test_thermal_budget_is_monotonic_decreasing():
    pc = PulseController(PulseConfig(temp_floor=60.0, temp_ceiling=80.0))
    temps = [60, 62, 65, 70, 75, 79, 80]
    budgets = [pc.thermal_budget(t) for t in temps]
    assert budgets == sorted(budgets, reverse=True)


def test_effective_burst_scales_with_budget():
    cfg = PulseConfig(temp_floor=60.0, temp_ceiling=80.0,
                      min_burst_ms=5.0, max_burst_ms=200.0)
    pc = PulseController(cfg)
    # At floor: full budget -> max burst.
    assert math.isclose(pc.effective_burst_ms(60.0), cfg.max_burst_ms)
    # At ceiling: zero budget -> min burst.
    assert math.isclose(pc.effective_burst_ms(80.0), cfg.min_burst_ms)
    # Hotter never bursts longer than cooler.
    assert pc.effective_burst_ms(75.0) < pc.effective_burst_ms(65.0)


def test_effective_cooldown_grows_when_hot():
    cfg = PulseConfig(temp_floor=60.0, temp_ceiling=80.0, cooldown_ms=10.0)
    pc = PulseController(cfg)
    cool_at_floor = pc.effective_cooldown_ms(60.0)
    cool_at_ceiling = pc.effective_cooldown_ms(80.0)
    # Full budget -> base cooldown; zero budget -> 3x base cooldown.
    assert math.isclose(cool_at_floor, cfg.cooldown_ms)
    assert math.isclose(cool_at_ceiling, cfg.cooldown_ms * 3.0)
    assert cool_at_ceiling > cool_at_floor


def test_should_not_fire_at_ceiling():
    pc = PulseController(PulseConfig(temp_ceiling=80.0))
    assert pc.should_fire_gpu(80.0) is False
    assert pc.should_fire_gpu(90.0) is False


def test_duty_cycle_empty_is_zero():
    pc = PulseController()
    assert pc.duty_cycle == 0.0


def test_duty_cycle_accumulates():
    pc = PulseController()
    pc.record_burst(30.0)
    pc.record_cooldown(10.0)
    # 30 / (30 + 10) = 0.75
    assert math.isclose(pc.duty_cycle, 0.75)
    stats = pc.stats
    assert stats["burst_count"] == 1
    assert math.isclose(stats["total_burst_ms"], 30.0)
    assert math.isclose(stats["duty_cycle"], 0.75)
