"""Timing utilities for pricing oracle instrumentation."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CallRecord:
    """Timing record for a single oracle.solve() invocation."""

    api_seconds: float = 0.0
    extract_seconds: float = 0.0
    columns_found: int = 0


class OracleTimer:
    """Accumulates per-call timing records for a pricing oracle.

    Usage::

        timer = OracleTimer()
        # inside oracle.solve():
        t0 = time.time(); response = solver.solve(...); api_s = time.time() - t0
        t0 = time.time(); cols = extract(...); ext_s = time.time() - t0
        timer.record(api_seconds=api_s, extract_seconds=ext_s, columns_found=len(cols))

        # after CG run:
        print(timer.summary())
    """

    def __init__(self) -> None:
        self.calls: List[CallRecord] = []

    def record(
        self,
        api_seconds: float = 0.0,
        extract_seconds: float = 0.0,
        columns_found: int = 0,
    ) -> None:
        self.calls.append(CallRecord(
            api_seconds=api_seconds,
            extract_seconds=extract_seconds,
            columns_found=columns_found,
        ))

    def reset(self) -> None:
        self.calls.clear()

    @property
    def num_calls(self) -> int:
        return len(self.calls)

    @property
    def total_api_seconds(self) -> float:
        return sum(c.api_seconds for c in self.calls)

    @property
    def total_extract_seconds(self) -> float:
        return sum(c.extract_seconds for c in self.calls)

    @property
    def total_columns_found(self) -> int:
        return sum(c.columns_found for c in self.calls)

    @property
    def avg_api_seconds(self) -> float:
        return self.total_api_seconds / self.num_calls if self.calls else 0.0

    @property
    def avg_columns_per_call(self) -> float:
        return self.total_columns_found / self.num_calls if self.calls else 0.0

    def summary(self) -> Dict[str, float]:
        """Return a dict of timing statistics suitable for JSON serialization."""
        return {
            "num_api_calls": self.num_calls,
            "total_api_seconds": round(self.total_api_seconds, 2),
            "total_extract_seconds": round(self.total_extract_seconds, 4),
            "avg_api_seconds": round(self.avg_api_seconds, 2),
            "avg_columns_per_call": round(self.avg_columns_per_call, 2),
            "total_columns_found": self.total_columns_found,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"OracleTimer({s['num_api_calls']} calls, "
            f"api={s['total_api_seconds']}s, "
            f"extract={s['total_extract_seconds']}s, "
            f"cols={s['total_columns_found']})"
        )
