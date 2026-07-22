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
2. the **fields from your output schema**, and
3. a **`status`** field.

For a schema extracting `product_name` and `price` from a CSV with `id` and `text`:

```json
[
  {
    "id": 1,
    "text": "A 250 ml bottle of olive oil for €4.99.",
    "token_count": 18,
    "product_name": "Olive oil 250ml",
    "price": 4.99,
    "status": "success"
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
| `"failure"` | The response could **not** be coerced into the schema, even after automatic fix-up attempts. |

On failure, the schema fields are still present but filled with **type-appropriate defaults** — empty string for `str`, `0` for `int`, `0.0` for `float`, `False` for `bool`, `[]` for lists, `{}` for dicts, `None` for optionals, and a random valid choice for `Literal`. This keeps every record the same shape, so downstream code doesn't break — but it means a `0.0` price could be a real zero *or* a failed extraction. **Always check `status`** before trusting a value.

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
