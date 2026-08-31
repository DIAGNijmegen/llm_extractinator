#!/usr/bin/env python
"""Diff two dev-test reports.

    python devtests/compare.py devtests/reports/A.json devtests/reports/B.json

This is what makes the harness useful for a decision rather than a glance: two
models against each other, or the same model before and after a change. It shows
movement, not verdicts — a fill rate that drops by 0.02 is noise, one that drops
by 0.3 is a regression, and only you can tell which from the numbers plus a look
at the actual output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


def _index(report: dict) -> dict:
    return {(r["model"], r["probe"]): r for r in report["results"]}


def _delta(before, after, digits: int = 3) -> str:
    if before is None or after is None:
        return "     n/a"
    change = after - before
    if isinstance(before, int) and isinstance(after, int):
        return f"{change:+d}" if change else "  ="
    return f"{change:+.{digits}f}" if abs(change) > 10 ** -digits else "  ="


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    before = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    after = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    old, new = _index(before), _index(after)

    shared = [key for key in new if key in old]
    if not shared:
        print("These reports have no model/probe in common — nothing to compare.")
        return 2

    header = (
        f"{'model':<14}{'probe':<11}{'ok':>10}{'fill':>12}"
        f"{'ratio':>12}{'num_ctx':>12}{'secs':>10}"
    )
    print(header)
    print("-" * len(header))
    for key in sorted(shared):
        a, b = old[key], new[key]
        model, probe = key
        moved = f"{a['ok']}→{b['ok']}"
        print(
            f"{model:<14}{probe:<11}{moved:>10}"
            f"{_delta(a['fill_rate'], b['fill_rate']):>12}"
            f"{_delta(a['calibration']['estimate_ratio'], b['calibration']['estimate_ratio']):>12}"
            f"{_delta(a['num_ctx'], b['num_ctx']):>12}"
            f"{_delta(a['seconds'], b['seconds'], 1):>10}"
        )

    print("\nonly in the first report: ", sorted(k for k in old if k not in new) or "none")
    print("only in the second report:", sorted(k for k in new if k not in old) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
