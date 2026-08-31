# Understanding output

After a run, LLM Extractinator writes your results to disk as JSON. This page explains **where** they go and **what's in them**.

---

## Where results land

Output is organised by run and by task:

```text
<output_dir>/<run_name>/<TaskName>-run<N>/nlp-predictions-dataset.json
```

With the defaults (`--output_dir output`, `--run_name run`), a run of `Task001.json` produces:

```text
output/
└── run/
    └── Task001-run0/
        └── nlp-predictions-dataset.json
```

- **`<run_name>`** comes from `--run_name` (default `run`) — use it to keep separate experiments apart.
- **`<TaskName>`** is the task file's name without extension (`Task001`, or `Task002_reports`).
- **`-run<N>`** counts repeats when you use `--n_runs > 1` (`-run0`, `-run1`, …).

Logs for the run are written under `--log_dir` (default `output/logs/`) in `task_runner.log`.

!!! note "Split and chunked runs"
    With `--max_context_len split`, short and long cases are processed separately and then **merged** back into a single `nlp-predictions-dataset.json`. With `--chunk_size`, the data is processed in batches and saved incrementally, then combined — so a crash only costs the current chunk.

---

## What a record looks like

The output file is a JSON list with **one object per input row**. Each object is the union of three things:

1. your **original input columns** (everything from the source row, plus an added `token_count`),
2. the **fields from your output schema**,
3. a **`status`** field, and
4. three **diagnostic** fields — `error_type`, `error_message` and `raw_output` — which are `null` on a successful row.

For a schema extracting `product_name` and `price` from a CSV with `id` and `text`:

```json
[
  {
    "id": 1,
    "text": "A 250 ml bottle of olive oil for €4.99.",
    "token_count": 18,
    "product_name": "Olive oil 250ml",
    "price": 4.99,
    "status": "success",
    "error_type": null,
    "error_message": null,
    "raw_output": null
  }
]
```

Because your original columns are carried through, you can join results straight back to your source data.

---

## The `status` field

Every record is tagged:

| `status` | Meaning |
|---|---|
| `"success"` | The model's response validated cleanly against your `OutputParser` schema. |
| `"failure"` | The response could **not** be coerced into the schema. |

On failure, every schema field is `null`. The keys are still there, so each record keeps the same shape and downstream code doesn't break — but nothing is invented. A `0.0` price means the model really said zero; a failed row says `null` and nothing else.

!!! note "This changed in 0.7.0"
    Failed rows used to be filled with type-appropriate defaults — `""` for text, `0` for numbers, and a *random* valid choice for a fixed set of values. That made a failure indistinguishable from an answer: a failed heart-rate row contributed a real `0` to a mean. If you have code that reads values without checking `status`, it was silently wrong before and will now see `null`/`NaN` instead.

### Why a row failed

Three fields describe the failure, and the most useful is `raw_output` — what the model actually said:

| Field | Meaning |
|---|---|
| `error_type` | The exception class, e.g. `OutputParserException` for unparseable output |
| `error_message` | The full error text |
| `raw_output` | The model's own response, before parsing — `null` if the request never completed |

Truncated JSON in `raw_output` means the answer ran out of room; prose means the model ignored the schema. Those need opposite fixes, and you cannot tell them apart without the text.

A run with any failures also writes a **`failures.json`** next to the predictions, containing just the failed rows with their input and the model's output — the same information, without having to filter several thousand records to find it.

!!! warning "Failures are expected sometimes"
    A few failures in a large run are normal, especially with smaller models or messy input. If most records fail, that usually points at a schema or prompt problem — see [Troubleshooting](troubleshooting.md).

---

## Inspecting results

- **In the Studio**, the **Results** tab gives you success/failure counts, filtering, search, and a per-record detail view — the quickest way to eyeball quality.
- **Programmatically**, load the file with any JSON or dataframe tool:

    ```python
    import pandas as pd
    df = pd.read_json("output/run/Task001-run0/nlp-predictions-dataset.json")
    print(df["status"].value_counts())
    failures = df[df["status"] == "failure"]
    ```

---

## Next

- Turn the knobs that affect quality and speed in the [settings reference](settings-reference.md).
- Chase down errors in [Troubleshooting](troubleshooting.md).
