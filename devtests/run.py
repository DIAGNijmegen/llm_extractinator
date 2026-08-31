#!/usr/bin/env python
"""Run the dev-test suite against a real model.

    python devtests/run.py --model phi4
    python devtests/run.py --model phi4 --host http://localhost:11434
    python devtests/run.py --model qwen3:4b --model phi4
    python devtests/run.py --list

Without ``--host`` the tool manages Ollama itself, exactly as a normal run does:
it starts a server if one is not already up, pulls the model, and stops it
afterwards. With ``--host`` it only connects — nothing is started, pulled or
stopped — which is what you want against a shared or remote GPU box.

What this is for
----------------
The offline suite in ``tests/`` proves the plumbing with a faked model. These
tasks exist only for the questions a faked model cannot answer, because it
always returns whatever it was told to: does grammar-constrained decoding hold
the model to an enum, is the schema-derived output budget actually enough, does
the token estimate drift on Dutch, do field descriptions change the result.

It reports numbers rather than asserting them. A real model's output is not
stable enough for golden files, and an accuracy threshold in a test only trains
you to ignore a red suite. The exit code is non-zero only for hard breakage —
an exception, or a task where nothing at all parsed.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
TASKS_DIR = HERE / "tasks"
DATA_DIR = HERE / "data"
OUTPUT_DIR = HERE / "output"
REPORTS_DIR = HERE / "reports"

# name -> (task id, what only a real model can tell you)
PROBES = {
    "basic": (101, "control: does anything work at all"),
    "enums": (102, "does the grammar hold the model to the allowed values"),
    "wide": (103, "is the schema-derived num_predict actually enough (21 fields)"),
    "nested": (104, "list-of-objects under the grammar"),
    "optional": (105, "does it omit absent fields now that `required` is not forced"),
    "reasoning": (106, "format + think together; needs a judgement, not a copy"),
    "dutch": (107, "token-estimate drift on non-English text — read the ratio"),
    "long": (108, "context sizing on ~2k-token documents"),
    "described": (109, "A/B: fields WITH descriptions"),
    "bare": (110, "A/B: the same fields WITHOUT descriptions"),
}

_CONTEXT_RE = re.compile(r"context (\d+) = (\d+) prompt \+ (\d+) answer")


class _LogCapture(logging.Handler):
    """Keeps the run's log records so the numbers can be read back.

    The pipeline logs its budget and calibration with %-style arguments, so the
    values are available as ``record.args`` without parsing prose. This is a
    deliberate coupling to log messages — acceptable for a dev harness, but if a
    column reads ``n/a`` and you did not expect it, a message was probably
    reworded.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def starting(self, prefix: str) -> List[logging.LogRecord]:
        return [
            r for r in self.records
            if isinstance(r.msg, str) and r.msg.startswith(prefix)
        ]


def _read_budget(capture: _LogCapture) -> Dict[str, Optional[int]]:
    for record in capture.starting("Budget"):
        match = _CONTEXT_RE.search(str(record.args[-1]))
        if match:
            ctx, prompt, answer = (int(g) for g in match.groups())
            return {"num_ctx": ctx, "prompt_reserved": prompt, "num_predict": answer}
    return {"num_ctx": None, "prompt_reserved": None, "num_predict": None}


def _read_calibration(capture: _LogCapture) -> Dict[str, Optional[float]]:
    for record in capture.starting("Prompt calibration:"):
        reserved, measured, ratio, _spare = record.args
        return {
            "reserved": int(reserved),
            "measured": int(measured),
            "estimate_ratio": round(float(ratio), 3),
        }
    return {"reserved": None, "measured": None, "estimate_ratio": None}


def _fill_rate(rows: List[dict], schema_fields: List[str]) -> Optional[float]:
    """Of the values a successful row could have carried, how many are present.

    A crude proxy for extraction quality, but the right shape for the A/B: two
    runs over the same data with the same schema differ only in how much the
    model actually found.
    """
    successes = [r for r in rows if r.get("status") == "success"]
    if not successes or not schema_fields:
        return None
    filled = sum(
        1
        for row in successes
        for name in schema_fields
        if row.get(name) not in (None, "", [], {})
    )
    return round(filled / (len(successes) * len(schema_fields)), 3)


def run_probe(
    name: str, model: str, host: Optional[str], limit: Optional[int]
) -> dict:
    """Run one probe and collect what it tells us."""
    from llm_extractinator import extractinate

    task_id, purpose = PROBES[name]
    task_file = next(TASKS_DIR.glob(f"Task{task_id:03d}_*.json"))
    task_config = json.loads(task_file.read_text(encoding="utf-8"))
    schema_fields = list(task_config.get("Parser_Format", {}))

    run_name = f"{name}"
    out_dir = OUTPUT_DIR / model.replace(":", "_").replace("/", "_")

    capture = _LogCapture()
    root = logging.getLogger()
    previous_level = root.level

    # The level has to be set here, explicitly. ``extractinate`` configures
    # logging through ``logging.basicConfig``, which is a no-op once the root
    # logger already has a handler — and attaching the capture below gives it
    # one. Without this the root stays at WARNING, every INFO record is dropped
    # before any handler sees it, and every number this harness reports comes
    # back "n/a" while the run itself works perfectly.
    root.setLevel(logging.INFO)
    root.addHandler(capture)

    # basicConfig being a no-op also means the run's own log file is never
    # created, so one is attached per probe. When a probe misbehaves this is
    # where the budget, the calibration and the failure summary are written out
    # in full.
    log_path = out_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = logging.FileHandler(log_path, encoding="utf-8")
    log_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(log_file)

    started = time.time()
    error = None
    try:
        extractinate(
            task_id=task_id,
            model_name=model,
            ollama_host=host,
            run_name=run_name,
            task_dir=TASKS_DIR,
            data_dir=DATA_DIR,
            example_dir=DATA_DIR,
            output_dir=out_dir,
            translation_dir=out_dir / "translations",
            log_dir=out_dir / "logs",
            test_run_size=limit,
            overwrite=True,
            seed=42,
        )
    except Exception as exc:  # a probe that blows up is a result, not a crash
        error = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed = time.time() - started
        root.removeHandler(capture)
        root.removeHandler(log_file)
        log_file.close()
        root.setLevel(previous_level)

    folder = out_dir / run_name
    predictions = sorted(folder.glob("*/nlp-predictions-dataset.json"))
    rows: List[dict] = []
    if predictions:
        rows = json.loads(predictions[0].read_text(encoding="utf-8"))

    failures = [r for r in rows if r.get("status") == "failure"]
    result = {
        "probe": name,
        "purpose": purpose,
        "task": task_file.stem,
        "model": model,
        "rows": len(rows),
        "ok": len(rows) - len(failures),
        "failed": len(failures),
        "failure_types": dict(Counter(r.get("error_type") for r in failures)),
        "fill_rate": _fill_rate(rows, schema_fields),
        "seconds": round(elapsed, 1),
        "error": error,
    }
    result.update(_read_budget(capture))
    result["calibration"] = _read_calibration(capture)
    return result


def _print_table(results: List[dict]) -> None:
    header = (
        f"{'probe':<11}{'rows':>5}{'ok':>5}{'fail':>5}"
        f"{'num_ctx':>9}{'n_pred':>8}{'measured':>9}{'ratio':>7}"
        f"{'fill':>7}{'secs':>7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        cal = r["calibration"]
        print(
            f"{r['probe']:<11}{r['rows']:>5}{r['ok']:>5}{r['failed']:>5}"
            f"{_n(r['num_ctx']):>9}{_n(r['num_predict']):>8}"
            f"{_n(cal['measured']):>9}{_n(cal['estimate_ratio']):>7}"
            f"{_n(r['fill_rate']):>7}{r['seconds']:>7}"
        )
    print()
    for r in results:
        if r["error"]:
            print(f"  !! {r['probe']}: {r['error']}")
        elif r["failure_types"]:
            kinds = ", ".join(f"{k} x{v}" for k, v in r["failure_types"].items())
            print(f"  ~  {r['probe']}: {kinds} — see failures.json in the run folder")


def _n(value) -> str:
    return "n/a" if value is None else str(value)


def _summarise(results: List[dict]) -> None:
    described = next((r for r in results if r["probe"] == "described"), None)
    bare = next((r for r in results if r["probe"] == "bare"), None)
    if described and bare and described["fill_rate"] and bare["fill_rate"]:
        delta = described["fill_rate"] - bare["fill_rate"]
        print(
            f"\n  Field descriptions: fill rate {bare['fill_rate']} without -> "
            f"{described['fill_rate']} with ({delta:+.3f}). "
            f"Read a sample of both before drawing a conclusion."
        )

    dutch = next((r for r in results if r["probe"] == "dutch"), None)
    basic = next((r for r in results if r["probe"] == "basic"), None)
    ratios = {
        r["probe"]: r["calibration"]["estimate_ratio"]
        for r in (basic, dutch)
        if r and r["calibration"]["estimate_ratio"]
    }
    if ratios:
        print(
            f"  Estimate/actual ratio: "
            + ", ".join(f"{k} {v}" for k, v in ratios.items())
            + ". Near 1.0 means the token estimate is sound; a large gap on "
            "Dutch is the argument for reading the real vocabulary."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-m", "--model", action="append", default=[], metavar="NAME",
        help="Model to run, Ollama naming. Repeat to sweep several.",
    )
    parser.add_argument(
        "--host", default=None, metavar="URL",
        help="Base URL of an already-running Ollama server, e.g. "
        "http://localhost:11434. Omit to let the tool manage its own.",
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated probe names to run (default: all). See --list.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Use only the first N rows of each dataset — a quick shake-down.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List the probes and what each is for."
    )
    parser.add_argument(
        "--keep-output", action="store_true",
        help="Keep any previous run output instead of clearing it first.",
    )
    args = parser.parse_args()

    if args.list:
        width = max(len(n) for n in PROBES)
        for name, (task_id, purpose) in PROBES.items():
            print(f"  {name:<{width}}  Task{task_id:03d}  {purpose}")
        return 0

    if not args.model:
        parser.error("at least one --model is required (try --model phi4)")

    names = list(PROBES)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in names if n not in PROBES]
        if unknown:
            parser.error(f"unknown probe(s): {', '.join(unknown)}. Try --list.")

    if not args.keep_output and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    where = args.host or "a server this run manages itself"
    print(f"Running {len(names)} probe(s) against {where}")

    all_results: List[dict] = []
    for model in args.model:
        print(f"\n=== {model} ===")
        results = []
        for name in names:
            print(f"  {name} ...", flush=True)
            results.append(run_probe(name, model, args.host, args.limit))
        _print_table(results)
        _summarise(results)
        all_results.extend(results)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report = REPORTS_DIR / f"{stamp}.json"
    report.write_text(
        json.dumps(
            {"when": stamp, "host": args.host, "results": all_results}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\nReport written to {report}")
    print(f"Compare two reports with: python devtests/compare.py A.json B.json")

    broken = [r for r in all_results if r["error"] or (r["rows"] and r["ok"] == 0)]
    if broken:
        print(f"\n{len(broken)} probe(s) failed outright: "
              f"{', '.join(r['probe'] for r in broken)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
