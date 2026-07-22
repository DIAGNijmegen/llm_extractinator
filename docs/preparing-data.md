# Preparing data

Your dataset is the text you want to extract *from*. LLM Extractinator processes it **one row (CSV) or one item (JSON) at a time**, reading a single text field from each.

You tell it which field to read in the task JSON, via `Input_Field`.

---

## Accepted formats

Datasets can be **CSV** or **JSON**. Each has one field holding the text; other columns are carried through untouched into the output.

**CSV** (`data/reports.csv`):

```text
id,text
1,"This is the first report..."
2,"This is the second report..."
```

**JSON** (`data/reports.json`) — a list of objects:

```json
[
  { "id": 1, "text": "This is the first report..." },
  { "id": 2, "text": "This is the second report..." }
]
```

In both cases `text` is the field we'll read. You can name it anything (`report`, `body`, `note`) as long as `Input_Field` matches it exactly.

!!! tip "Extra columns are kept"
    Any other columns — IDs, dates, metadata — pass straight through to each output record, so you can line results back up with your source data.

---

## Pointing the task at your data

In the task JSON:

```json
{
  "Data_Path": "reports.csv",
  "Input_Field": "text"
}
```

- **`Data_Path`** is relative to your data directory (`--data_dir`, default `data/`). So `reports.csv` means `data/reports.csv`.
- **`Input_Field`** must match the column/key name **exactly**, including case. A mismatch is one of the most common errors — the run stops with `'<name>' column not found`.

---

## A note on long text

Very long inputs cost more tokens and run slower. LLM Extractinator sizes the context window automatically (`--max_context_len max`), but if your documents vary a lot in length, `--max_context_len split` runs the short and long ones separately with right-sized windows. For very large datasets, `--chunk_size` processes them in batches and saves incrementally, so a crash only loses the current chunk. See the [settings reference](settings-reference.md).

---

## Few-shot examples (optional)

Examples are a *separate* file — not part of your dataset — used to show the model a few input→output pairs before it sees the real thing. They can noticeably improve output quality and consistency.

Because examples are their own topic (with their own required format), they have a dedicated page: **[Few-shot prompting](examples.md)**.

---

## Next

With your data ready, define the **[output schema](parser.md)** — the shape you want back.
