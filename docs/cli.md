# CLI usage

The command-line interface is `extractinate`. It runs a task file that already exists (build one in the [Studio](studio.md) or [by hand](manual-configuration.md)), which makes it ideal for repeatable and unattended runs.

Basic run:

```bash
extractinate --task_id 1 --model_name "phi4"
```

`--task_id 1` selects `Task001*.json` from your task directory. On first use, Ollama pulls the model automatically.

---

## The essentials

| Flag | Default | Purpose |
|---|---|---|
| `--task_id` | *required* | Which task to run (integer ID from the filename) |
| `--model_name` | `phi4` | Ollama model to use |
| `--run_name` | `run` | Names the output subfolder — keep experiments apart |
| `--num_examples` | `0` | Turn on few-shot prompting (needs `Example_Path` in the task) |
| `--reasoning_model` | off | Required for "thinking" models (see below) |
| `--verbose` | off | Stream full logs — the first thing to reach for when debugging |

Directories default to `data/`, `tasks/`, `examples/`, and `output/` in the current folder; override any of them (`--data_dir`, `--task_dir`, …) to point elsewhere. The **[settings reference](settings-reference.md)** documents every flag and every task-file field.

---

## Reasoning models

Models that emit intermediate reasoning before their answer (DeepSeek-R1, Qwen3, and similar) need `--reasoning_model` so the tool pulls the final JSON out of the surrounding thought process:

```bash
extractinate --task_id 1 --model_name "deepseek-r1" --reasoning_model
```

These models also need more room to generate — if output comes back empty or truncated, raise `--num_predict`.

---

## A few common recipes

**Reproducible run** (fixed seed, deterministic sampling):

```bash
extractinate --task_id 1 --model_name "phi4" --seed 42 --temperature 0.0
```

**Few-shot with a custom embedding model:**

```bash
extractinate --task_id 1 --num_examples 5 --embedding_model "mxbai-embed-large"
```

**Large dataset, saved incrementally** (crash-resistant):

```bash
extractinate --task_id 1 --model_name "phi4" --chunk_size 500
```

**Widely varying document lengths** (right-size the context per subset):

```bash
extractinate --task_id 1 --model_name "phi4" --max_context_len split
```

---

## What you get

Results are written to `output/<run_name>/<TaskName>-run<N>/nlp-predictions-dataset.json`, with logs under `output/logs/`. Each record carries your input columns, the extracted fields, and a `status`. See [Understanding output](output.md) — and always spot-check.
