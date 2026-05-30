"""Hardware personality — SQLite-backed learning from per-run metrics.

Each run records device, operation, duration, temperature, power. Over time
the personality builds a model of which device is best for which operation
under which conditions.
"""

import json
import sqlite3
import time
from pathlib import Path


DB_PATH = Path.home() / ".rag-race-router" / "personality.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    device TEXT NOT NULL,        -- 'cpu', 'gpu', 'npu'
    operation TEXT NOT NULL,     -- 'matmul', 'attention', 'layernorm', etc.
    duration_ms REAL NOT NULL,
    input_size INTEGER,         -- flattened input element count
    temp_before REAL,
    temp_after REAL,
    power_w REAL,
    success INTEGER DEFAULT 1,
    metadata TEXT                -- JSON blob for extras
);

CREATE TABLE IF NOT EXISTS routing_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,
    preferred_device TEXT NOT NULL,
    condition TEXT,              -- JSON: {"temp_below": 75, "input_size_below": 1000000}
    confidence REAL DEFAULT 0.5,
    sample_count INTEGER DEFAULT 0,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_device_op ON runs(device, operation);
CREATE INDEX IF NOT EXISTS idx_rules_op ON routing_rules(operation);
"""


class Personality:
    """Learns device preferences from runtime metrics."""

    def __init__(self, db_path: Path = None):
        self._db_path = db_path or DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def record_run(
        self,
        device: str,
        operation: str,
        duration_ms: float,
        input_size: int = None,
        temp_before: float = None,
        temp_after: float = None,
        power_w: float = None,
        success: bool = True,
        metadata: dict = None,
    ):
        self._conn.execute(
            """INSERT INTO runs
               (timestamp, device, operation, duration_ms, input_size,
                temp_before, temp_after, power_w, success, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), device, operation, duration_ms, input_size,
                temp_before, temp_after, power_w, int(success),
                json.dumps(metadata) if metadata else None,
            ),
        )
        self._conn.commit()

    def suggest(self, operation: str, temp: float = None, input_size: int = None) -> str:
        """Suggest the best device for an operation given conditions.

        Returns 'cpu', 'gpu', or 'npu'. Falls back to 'cpu' if no data.
        """
        # Check routing rules first
        rules = self._conn.execute(
            "SELECT * FROM routing_rules WHERE operation = ? ORDER BY confidence DESC",
            (operation,),
        ).fetchall()

        for rule in rules:
            condition = json.loads(rule["condition"]) if rule["condition"] else {}
            if temp and "temp_below" in condition and temp >= condition["temp_below"]:
                continue
            if input_size and "input_size_below" in condition and input_size >= condition["input_size_below"]:
                continue
            if rule["confidence"] >= 0.6 and rule["sample_count"] >= 5:
                return rule["preferred_device"]

        # Fall back to historical average
        rows = self._conn.execute(
            """SELECT device, AVG(duration_ms) as avg_ms, COUNT(*) as cnt
               FROM runs WHERE operation = ? AND success = 1
               GROUP BY device ORDER BY avg_ms ASC""",
            (operation,),
        ).fetchall()

        if rows and rows[0]["cnt"] >= 3:
            return rows[0]["device"]

        return "cpu"

    def thermal_budget(self) -> float:
        """Average power-normalized throughput across recent runs."""
        rows = self._conn.execute(
            """SELECT AVG(duration_ms) as avg_ms, AVG(power_w) as avg_w
               FROM runs WHERE timestamp > ? AND success = 1""",
            (time.time() - 300,),  # last 5 minutes
        ).fetchone()

        if rows and rows["avg_ms"] and rows["avg_w"] and rows["avg_w"] > 0:
            # Lower is better: ms per watt
            return rows["avg_ms"] / rows["avg_w"]
        return 0.0

    def update_rules(self, min_samples: int = 10):
        """Rebuild routing rules from accumulated run data."""
        operations = self._conn.execute(
            "SELECT DISTINCT operation FROM runs WHERE success = 1"
        ).fetchall()

        for row in operations:
            op = row["operation"]
            stats = self._conn.execute(
                """SELECT device, AVG(duration_ms) as avg_ms,
                          MIN(duration_ms) as min_ms, COUNT(*) as cnt
                   FROM runs WHERE operation = ? AND success = 1
                   GROUP BY device ORDER BY avg_ms ASC""",
                (op,),
            ).fetchall()

            if not stats or stats[0]["cnt"] < min_samples:
                continue

            best = stats[0]
            total_runs = sum(s["cnt"] for s in stats)
            confidence = min(1.0, best["cnt"] / max(total_runs, 1))

            # Upsert rule
            existing = self._conn.execute(
                "SELECT id FROM routing_rules WHERE operation = ? AND condition IS NULL",
                (op,),
            ).fetchone()

            if existing:
                self._conn.execute(
                    """UPDATE routing_rules
                       SET preferred_device = ?, confidence = ?,
                           sample_count = ?, updated_at = ?
                       WHERE id = ?""",
                    (best["device"], confidence, best["cnt"], time.time(), existing["id"]),
                )
            else:
                self._conn.execute(
                    """INSERT INTO routing_rules
                       (operation, preferred_device, confidence, sample_count, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (op, best["device"], confidence, best["cnt"], time.time()),
                )

        self._conn.commit()

    def show(self) -> dict:
        """Return personality summary."""
        total_runs = self._conn.execute("SELECT COUNT(*) as c FROM runs").fetchone()["c"]
        by_device = self._conn.execute(
            """SELECT device, COUNT(*) as cnt, AVG(duration_ms) as avg_ms
               FROM runs WHERE success = 1 GROUP BY device"""
        ).fetchall()
        rules = self._conn.execute(
            "SELECT operation, preferred_device, confidence, sample_count FROM routing_rules"
        ).fetchall()

        return {
            "total_runs": total_runs,
            "by_device": {
                r["device"]: {"count": r["cnt"], "avg_ms": round(r["avg_ms"], 2)}
                for r in by_device
            },
            "routing_rules": [
                {
                    "operation": r["operation"],
                    "preferred_device": r["preferred_device"],
                    "confidence": round(r["confidence"], 3),
                    "samples": r["sample_count"],
                }
                for r in rules
            ],
        }

    def show_table(self) -> str:
        """Return a formatted table of the personality profile."""
        data = self.show()
        total = data["total_runs"]
        rules = data["routing_rules"]
        by_device = data["by_device"]

        if not rules and total == 0:
            return "Hardware Personality: No data yet. Run --demo to build profile."

        lines = [f"Hardware Personality ({total} runs):"]

        # Reroute counts from runs metadata
        reroute_rows = self._conn.execute(
            """SELECT operation, COUNT(*) as cnt
               FROM runs WHERE metadata LIKE '%reroute%'
               GROUP BY operation"""
        ).fetchall()
        reroutes = {r["operation"]: r["cnt"] for r in reroute_rows}

        if rules:
            # Header
            lines.append(
                f"{'Operation':<16} {'Best on':<8} {'Avg (ms)':<10} {'Samples':<9} {'Conf':<6} {'Reroutes'}"
            )
            lines.append("-" * 65)

            for rule in rules:
                op = rule["operation"]
                dev = rule["preferred_device"]
                conf = rule["confidence"]
                samples = rule["samples"]

                # Get avg_ms for best device
                row = self._conn.execute(
                    "SELECT AVG(duration_ms) as avg FROM runs WHERE operation=? AND device=? AND success=1",
                    (op, dev),
                ).fetchone()
                avg = round(row["avg"], 2) if row and row["avg"] else 0.0

                rr = reroutes.get(op, 0)
                rr_str = f"{rr} (heat)" if rr > 0 else "0"
                lines.append(
                    f"{op:<16} {dev:<8} {avg:<10.2f} {samples:<9} {conf:<6.2f} {rr_str}"
                )
        else:
            lines.append("  No routing rules yet — need more runs.")

        if by_device:
            lines.append("")
            lines.append("By device:")
            for dev, info in by_device.items():
                lines.append(f"  {dev}: {info['count']} runs, avg {info['avg_ms']:.2f}ms")

        return "\n".join(lines)

    def close(self):
        self._conn.close()
