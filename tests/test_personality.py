"""Unit tests for the SQLite-backed routing learner in engine/personality.py.

Uses a throwaway database file under a pytest tmp_path — no real hardware,
no global state, no network.
"""

import pytest

from conftest import load_module

personality = load_module("personality")
Personality = personality.Personality


@pytest.fixture
def db(tmp_path):
    p = Personality(db_path=tmp_path / "personality.db")
    yield p
    p.close()


def test_suggest_defaults_to_cpu_with_no_data(db):
    assert db.suggest("matmul") == "cpu"


def test_record_run_increments_total(db):
    assert db.show()["total_runs"] == 0
    db.record_run(device="gpu", operation="matmul", duration_ms=1.0)
    db.record_run(device="cpu", operation="matmul", duration_ms=5.0)
    assert db.show()["total_runs"] == 2


def test_suggest_picks_fastest_device_after_enough_history(db):
    # GPU is consistently faster than CPU for matmul; need >= 3 samples
    # on the winning device for the historical fallback to fire.
    for _ in range(4):
        db.record_run(device="gpu", operation="matmul", duration_ms=1.0)
        db.record_run(device="cpu", operation="matmul", duration_ms=10.0)
    assert db.suggest("matmul") == "gpu"


def test_suggest_ignores_failed_runs(db):
    # Failures must not count toward the historical average.
    for _ in range(5):
        db.record_run(device="gpu", operation="conv", duration_ms=1.0, success=False)
    # No successful runs -> still defaults to cpu.
    assert db.suggest("conv") == "cpu"


def test_update_rules_encodes_preferred_device(db):
    # Need >= min_samples (default 10) successful runs to encode a rule.
    for _ in range(12):
        db.record_run(device="npu", operation="embed", duration_ms=0.5)
        db.record_run(device="cpu", operation="embed", duration_ms=4.0)
    db.update_rules()
    rules = {r["operation"]: r for r in db.show()["routing_rules"]}
    assert "embed" in rules
    assert rules["embed"]["preferred_device"] == "npu"
    assert rules["embed"]["samples"] >= 10
    assert 0.0 <= rules["embed"]["confidence"] <= 1.0


def test_update_rules_skips_underpopulated_ops(db):
    # Only a few samples -> no rule should be created.
    for _ in range(3):
        db.record_run(device="gpu", operation="rare_op", duration_ms=1.0)
    db.update_rules()
    ops_with_rules = {r["operation"] for r in db.show()["routing_rules"]}
    assert "rare_op" not in ops_with_rules


def test_show_groups_by_device(db):
    db.record_run(device="gpu", operation="matmul", duration_ms=2.0)
    db.record_run(device="gpu", operation="matmul", duration_ms=4.0)
    db.record_run(device="cpu", operation="tokenize", duration_ms=1.0)
    by_device = db.show()["by_device"]
    assert by_device["gpu"]["count"] == 2
    assert by_device["gpu"]["avg_ms"] == 3.0
    assert by_device["cpu"]["count"] == 1


def test_show_table_handles_empty_db(db):
    text = db.show_table()
    assert "No data yet" in text
