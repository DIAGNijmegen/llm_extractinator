import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Grammar of a per-task-run output folder: "<task_name>[-test<N>]-run<idx>".
# The writer (PredictionTask) and the reader (Studio's Results tab) both go
# through the two helpers below so the two can never drift apart.
_TEST_RUN_RE = re.compile(r"-test(\d+)-run\d+$")


def run_folder_name(
    task_name: str, run_idx: int, test_run_size: Optional[int] = None
) -> str:
    """Build the output folder name for one run of a task.

    The row count is part of the name for a test run so that repeating one at a
    different size resolves to a different folder — otherwise the
    skip-if-exists guard would hand back an earlier, smaller run's output.
    """
    suffix = f"-test{test_run_size}" if test_run_size else ""
    return f"{task_name}{suffix}-run{run_idx}"


def parse_test_run_size(run_name: str) -> Optional[int]:
    """Return the requested row count if ``run_name`` is a test run, else None."""
    match = _TEST_RUN_RE.search(run_name)
    return int(match.group(1)) if match else None


def chunk_files(output_path: Path) -> list:
    """Return this run's chunk files, ordered by the row offset in their name.

    Chunk files are ``nlp-predictions-dataset-<row offset>.json``. The digit
    filter keeps the ``-short``/``-long`` split files out, and the numeric sort
    matters because both glob order (arbitrary) and lexical order ("-100"
    before "-50") would scramble row order in the merged output.
    """
    return sorted(
        (
            f
            for f in output_path.glob("nlp-predictions-dataset-*.json")
            if f.stem.split("-")[-1].isdigit()
        ),
        key=lambda f: int(f.stem.split("-")[-1]),
    )


def save_json(
    data,
    outpath: Path,
    filename: Optional[str] = None,
    retries: int = 3,
    delay: float = 1.0,
):
    path = outpath / filename if filename else outpath
    if isinstance(data, BaseModel):
        data = data.model_dump()

    attempt = 0
    while attempt < retries:
        try:
            with path.open("w+") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Data saved to {path}.")
            break
        except IOError as e:
            attempt += 1
            logger.error(
                f"Failed to save data to {path}. Retrying ({attempt}/{retries})..."
            )
            time.sleep(delay)
    else:
        logger.error(f"Failed to save data to {path} after {retries} attempts.")


#: Enough of the input to recognise which document failed, without turning the
#: failures file into a second copy of the dataset.
FAILURE_INPUT_PREVIEW_CHARS = 500


def write_failures(
    rows: List[Dict],
    input_field: str,
    outpath: Path,
    data_split: Optional[str] = None,
) -> Optional[Path]:
    """Collect the failed rows into a file you can actually open.

    A handful of failures is easy to miss in a predictions file of several
    thousand rows, and the column that explains them — the model's own output —
    is the widest one there. Written only when something failed, so a clean run
    leaves no confusing empty artefact.

    Lives here rather than on ``PredictionTask`` because ``split`` writes one per
    phase and then has to replace them with a single merged file, so the runner
    needs it too.
    """
    failures = [
        {
            "row": index,
            "input": str(row.get(input_field))[:FAILURE_INPUT_PREVIEW_CHARS],
            "error_type": row.get("error_type"),
            "error_message": row.get("error_message"),
            "raw_output": row.get("raw_output"),
        }
        for index, row in enumerate(rows)
        if row.get("status") == "failure"
    ]
    if not failures:
        return None

    name = f"failures-{data_split}.json" if data_split else "failures.json"
    save_json(failures, outpath=outpath, filename=name)
    logger.warning(
        "%d failed row(s); the model's output for each is in %s",
        len(failures),
        outpath / name,
    )
    return outpath / name
