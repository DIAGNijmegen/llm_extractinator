# Troubleshooting

Common problems and how to fix them. If your issue isn't here, check the run log (`output/logs/task_runner.log`, or run with `--verbose`) and open an [issue on GitHub](https://github.com/DIAGNijmegen/llm_extractinator/issues).

---

## Setup and connection

### "Connection refused" / the tool can't reach Ollama

Ollama isn't running, or isn't on the expected port.

- On desktop, launch the Ollama app. On Linux, run `ollama serve`.
- Check it's up: `curl http://localhost:11434/api/tags` should return JSON.
- If Ollama runs elsewhere or on a non-default port, pass `--ollama_host http://host:port`.

### The model won't download or isn't found

- Confirm the name matches Ollama's naming exactly (e.g. `qwen3:8b`, not `qwen3-8b`). Browse [ollama.com/library](https://ollama.com/library).
- Let the tool pull it, or pre-pull with `ollama pull <model>`.
- **Using `--ollama_host`?** The tool will **not** pull models onto an externally managed server — the model must already exist there (`ollama pull <model>` on that machine).

---

## Task and file errors

### `No file found matching Task00X*.json`

The ID you passed doesn't match a task file. Files must be named `Task<NNN>...json` with a **zero-padded three-digit** ID, and you reference them by the integer:

- `Task001.json` or `Task001_products.json` → `--task_id 1`
- `Task012_reports.json` → `--task_id 12`

Also check you're pointing at the right folder with `--task_dir`.

### `Multiple files found matching Task00X*.json`

Two task files share the same ID (e.g. `Task001.json` and `Task001_old.json`). Give each a unique ID or move one out of `tasks/`.

### `'<name>' column not found in test cases`

`Input_Field` in the task JSON doesn't match a column/key in your dataset. It must match **exactly**, including case. Open the dataset and confirm the field name.

### `Examples must have 'input' and 'output' columns`

Your examples file uses the wrong field names. It must have exactly `input` and `output` — see [Few-shot prompting](examples.md). This is independent of your dataset's `Input_Field`.

### The output schema won't load

- The file must define a top-level class named exactly **`OutputParser`** that inherits from `BaseModel`.
- It must be valid Python — if you hand-edited it, try importing it: `python -c "import tasks.parsers.your_schema"`.
- It belongs in `tasks/parsers/`, and the task JSON references just the **filename** (`"Parser_Format": "your_schema.py"`).

---

## Quality issues

### Most records come back as `"failure"`

**Nine times out of ten this is the token limit.** When almost *everything* fails, the model is running out of output tokens before it can finish the JSON — the response is cut off mid-object, so it can't be parsed into your schema. Fix this first, before touching anything else:

1. **Raise `--num_predict`.** This caps how many tokens the model may generate per response (default `512`), and it's the usual culprit. Nested schemas, long field values, and reasoning models all need more room — try `1024`, `2048`, or higher:

    ```bash
    extractinate --task_id 1 --model_name "phi4" --num_predict 2048
    ```

2. **Using a reasoning model?** Its "thinking" is counted against that same output budget, so it needs *even more*. Make sure `--reasoning_model` is set **and** give `--num_predict` a generous value — this combination is the most common cause of all-failure runs.

If failures persist even with plenty of output room, then look at the prompt and schema:

3. **Simplify the schema** — start with two or three fields and expand once they're reliable.
4. **Use a stronger model** — small models struggle with complex or deeply nested schemas.
5. **Add `Field(description=...)`** to ambiguous fields, or sharpen the task `Description`.
6. **Add a few examples** with `--num_examples` (see [Few-shot prompting](examples.md)).

### Values look plausible but are wrong

This is the LLM hallucinating, not a bug. Lower `--temperature` to `0.0` for determinism, add examples, and **always spot-check** — especially numbers and anything clinical or high-stakes.

### Reasoning models return empty or malformed JSON

Models that "think" before answering (DeepSeek-R1, Qwen3, …) need `--reasoning_model` so the tool separates the chain-of-thought from the final JSON. Without it, the reasoning text can swamp the answer. They also need more output budget — raise `--num_predict`.

---

## Performance

### Runs are very slow

- You're likely on CPU or an oversized model — see [CPU-only hardware](cpu-only.md) and pick something smaller. A good small starting point is `qwen3.5:2b`; scale up only if quality needs it.
- Trim `--num_predict` and `--max_context_len`, and keep `--num_examples` low.
- For big datasets, `--chunk_size` saves incrementally so you can stop and resume.

### Out-of-memory / GPU errors

The model is too large for your hardware. Drop to a smaller model — `qwen3.5:2b` is a safe starting point — or a more heavily quantized variant, or run CPU-only. Run with `--verbose` to see Ollama's own error output.

---

## Still stuck?

Run with `--verbose` to get full logs, note the exact command and error, and open an [issue](https://github.com/DIAGNijmegen/llm_extractinator/issues) with those details.
