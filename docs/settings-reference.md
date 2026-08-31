# Settings & Flags Reference

This page provides a complete overview of all configuration options in **LLM Extractinator**.

It follows a professional documentation pattern:

1. **A quick summary table** for fast scanning  
2. **Detailed per‑flag descriptions** for deeper understanding

---

## 1. CLI Flags Overview (Summary)

| Flag | Default | Description |
|------|---------|-------------|
| `--task_id` | _required_ | Selects which task JSON file to run. |
| `--run_name` | `"run"` | Name used in logs and output folders. |
| `--n_runs` | `1` | Number of times to repeat the task. |
| `--verbose` | `False` | Enables detailed logging. |
| `--overwrite` | `False` | Overwrites existing outputs if enabled. |
| `--seed` | `None` | Random seed for reproducibility. |
| `--model_name` | `"phi4"` | Model used via Ollama. |
| `--ollama_host` | `None` | Connect to an already-running Ollama server instead of managing one. |
| `--embedding_model` | `"nomic-embed-text"` | Embedding model for few‑shot selection. |
| `--temperature` | *auto* | Sampling randomness; `0.0` normally, `0.6` for a thinking model. |
| `--top_k` | `None` | Top‑K sampling. |
| `--top_p` | `None` | Nucleus sampling. |
| `--num_predict` | *auto* | Maximum generated tokens; sized from the schema when unset. |
| `--max_context_len` | `"max"` | Context length strategy. |
| `--max_context_cap` | `None` | Upper bound on the context window in `max`/`split` mode. Must exceed `--num_predict`. |
| `--quantile` | `0.8` | Split point for `--max_context_len split` (short vs long cases). |
| `--reasoning_model` | `False` | Forces reasoning‑model mode on (thinking models are auto‑detected). |
| `--no_reasoning` | `False` | Forces reasoning‑model mode off, overriding auto‑detection. |
| `--num_examples` | `0` | Number of few‑shot examples. |
| `--chunk_size` | `None` | Chunk size for long inputs. |
| `--test_run_size` | `None` | Run on only the first N rows (or a random sample) of the test set. |
| `--test_run_random` | `False` | With `--test_run_size`, sample randomly instead of taking the first N rows. |
| `--translate` | `False` | Translate input to English first. |
| `--output_dir` | `output/` | Output location. |
| `--log_dir` | `output/` | Log location. |
| `--data_dir` | `data/` | Input data directory. |
| `--task_dir` | `tasks/` | Task JSON directory. |
| `--example_dir` | `examples/` | Few‑shot example directory. |
| `--translation_dir` | `translations/` | Translation output directory. |

---

## 2. Detailed CLI Flag Descriptions

### `--task_id`
**Type:** `int`  
**Default:** _required_  
Selects which task JSON file to run, based on its numeric prefix  
(e.g., `Task001_*.json` → `--task_id 1`).

---

### `--run_name`
**Type:** `str`  
**Default:** `"run"`  
Human‑friendly name used to structure log and output folders.

---

### `--n_runs`
**Type:** `int`  
**Default:** `1`  
Runs the same extraction multiple times—useful for testing stability or variance.

---

### `--verbose`
**Type:** `bool`  
**Default:** `False`  
Prints additional diagnostic information during execution.

---

### `--overwrite`
**Type:** `bool`  
**Default:** `False`  
If enabled, existing run results in the output folder will be overwritten. If disabled, the tool will skip processing if output already exists.

---

### `--seed`
**Type:** `int`  
**Default:** `None`  
Random seed for reproducible behavior where possible.

---

### `--model_name`

**Type:** `str`
**Default:** `"phi4"`
Name of the Ollama model to use (e.g., `"phi4"`, `"llama3.3"`, `"deepseek-r1:8b"`). See [Ollama models](https://ollama.com/models) for available options.

---

### `--ollama_host`

**Type:** `str`
**Default:** `None`
Base URL of an already-running Ollama server, e.g. `"http://localhost:11500"` or `"http://remote-machine:11434"`.

- **Unset (default):** llm_extractinator manages its own server — starting it, pulling `--model_name`/`--embedding_model` as needed, and stopping the model when the run finishes.
- **Set:** the server is treated as externally managed — llm_extractinator only connects to it. It will not start/stop the server and will not pull models onto it, so make sure the model is already available there.

---

### `--embedding_model`

**Type:** `str`
**Default:** `"nomic-embed-text"`
Name of the embedding model to use for few-shot example selection via semantic similarity. Only used when `--num_examples > 0`. See [Ollama models](https://ollama.com/models) for available embedding models (e.g., `"mxbai-embed-large"`, `"nomic-embed-text"`).

---

### `--temperature`

**Type:** `float`  
**Default:** chosen automatically  

Left unset, the temperature is picked from the model: **0.0** (greedy, deterministic) for an ordinary model, **0.6** for a thinking one.

That second case is not a preference. Qwen's model cards state it plainly for thinking mode: *"DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions."* A repetition loop is a particularly bad failure for extraction — the model spends its whole `--num_predict` budget looping and returns no JSON at all. Qwen recommends 0.6 alongside `--top_p 0.95` and `--top_k 20`; only the temperature is applied for you, so set those two yourself if you want the full recommended configuration.

Reproducibility is not lost: with `--seed`, a non-zero temperature is still deterministic run to run.

Pass a value to override, including `--temperature 0` to force greedy on a thinking model anyway — the run logs a warning when you do.

---

### `--top_k`
**Type:** `int`  
**Default:** `None`  
Restricts sampling to the top‑K highest‑probability tokens.

---

### `--top_p`
**Type:** `float`  
**Default:** `None`  
Nucleus sampling: sample from the smallest token set whose cumulative probability ≥ `p`.

---

### `--num_predict`
**Type:** `int`  
**Default:** *auto — sized from the output schema*  
Maximum number of tokens to generate for the model’s output.

Left unset, this is derived from your schema: a schema of thirty free-text
fields needs far more room than one of three enums, and a truncated answer loses
the whole row rather than degrading gracefully. The derived value never goes
below 512, so it can only ever raise the budget relative to the old fixed
default. A thinking model gets a further allowance on top, because its chain of
thought is charged to this same budget.

Set it explicitly to override. If output comes back truncated on a model that
reasons regardless of `think: false`, raising it is the fix.

---

### `--max_context_len`
**Type:** `str` or `int`  
**Default:** `"max"`  
Controls context length policy:
- `"max"` — use maximum available length  
- `"split"` — split dataset in two by input size, running each subset with a right-sized context (good when lengths vary a lot); results are merged afterwards  
- integer — explicitly set context length

---

### `--max_context_cap`
**Type:** `int`  
**Default:** `None`  
An upper bound on the context window when `--max_context_len` is `max` or `split`. The window is still fitted to your data; it just never grows past this value. Inputs longer than the cap are truncated.

It must be larger than `--num_predict`: `num_ctx` is the *whole* window, so the prompt and the generated answer share it, and a ceiling at or below the output budget would leave no room for your documents at all.

This is the knob for fitting a run into a given GPU. `max` sizes the window to the longest document in the dataset, which is right for accuracy but says nothing about what your card can hold — and the context window, not the model weights, is usually what turns a run that fits into one that runs out of memory. The Studio's hardware presets set this for you.

---

### `--quantile`
**Type:** `float`  
**Default:** `0.8`  
Only used with `--max_context_len split`. Sets the token-count quantile that divides "short" from "long" cases — at `0.8`, the longest 20% are run as the long subset. Must be between 0 and 1.

---

### `--reasoning_model`
**Type:** `bool`  
**Default:** `False`  
For models like DeepSeek‑R1 and Qwen3 that output chain‑of‑thought before JSON, so the reasoning is routed away from the answer instead of swamping it.

You usually don't need this flag: models already installed in Ollama are inspected for a `thinking` capability and reasoning mode is switched on automatically. Set it when the model has not been pulled yet, or when the server can't be reached, so auto‑detection can't see the model. It also raises the generation budget — see [`--num_predict`](#--num_predict).

---

### `--no_reasoning`
**Type:** `bool` (flag)  
**Default:** `False`  
Forces reasoning mode off. This overrides both auto‑detection and `--reasoning_model`, so it's the way to make a thinking model answer directly. Passing both flags together logs a warning and `--no_reasoning` wins.

On a model that advertises thinking, this sends Ollama an explicit `think: false`. On a model that doesn't, it does nothing at all — Ollama rejects `think` outright for those, and there is no reasoning to disable anyway.

A few models (qwen3.5 among them) reason regardless of `think: false`. Their reasoning is stripped from the output before parsing so extraction still works, but it is still generated, so it still spends the `--num_predict` budget. If output comes back truncated on such a model, either raise `--num_predict` or leave reasoning enabled.

---

### `--num_examples`
**Type:** `int`  
**Default:** `0`  
Number of few‑shot examples to include in the prompt.  
Requires setting `Example_Path` inside the task JSON file.

---

### `--chunk_size`
**Type:** `int`  
**Default:** `None`  
Splits the dataset into chunks of this many documents for processing. Useful for very large datasets as the chunks are saved incrementally. If a crash occurs, only the current chunk needs to be reprocessed.

---

### `--test_run_size`
**Type:** `int`  
**Default:** `None`  
Runs on only the first N rows of the test set instead of the full dataset — a quick sanity check before committing to a full run. The context window is still sized from the *full* dataset, so what the subset exercises matches what the real run will do. Combine with `--test_run_random` to sample randomly instead. Output is written to its own folder, `<task_name>-test<N>-run<idx>`, so it never collides with a full run's output — and because the row count is part of the name, repeating a test run at a different size writes somewhere new instead of silently returning the earlier, smaller run's results. Not compatible with `--max_context_len split`, which falls back to `max` automatically.

---

### `--test_run_random`
**Type:** `bool` (flag)  
**Default:** `False`  
With `--test_run_size`, sample rows randomly instead of taking the first N. Reproducible via `--seed`.

---

### `--translate`
**Type:** `bool`  
**Default:** `False`  
If enabled, input is translated to English before extraction—adds an extra model step. **Not recommended!**

---

### `--output_dir`
**Type:** `Path`  
**Default:** `output/`  
Where extracted results are written.

---

### `--log_dir`
**Type:** `Path`  
**Default:** `output/`  
Location for logs; defaults to the output directory.

---

### `--data_dir`
**Type:** `Path`  
**Default:** `data/`  
Directory containing datasets referenced by the `Data_Path` in task JSON files.

---

### `--task_dir`
**Type:** `Path`  
**Default:** `tasks/`  
Folder containing task JSON files.

---

### `--example_dir`
**Type:** `Path`  
**Default:** `examples/`  
Directory referenced by `Example_Path` in task JSON files.

---

### `--translation_dir`
**Type:** `Path`  
**Default:** `translations/`  
Folder where translated versions of inputs are saved when using `--translate`.

---

## 3. Task Configuration Files

Task files define what to extract and how to parse it. They must be named:

```
Task<NNN>...json
```

where `<NNN>` is a **zero-padded three-digit ID**. Anything may follow the ID (an optional `_name` is common) as long as it isn't another digit, so all of these are valid and share ID `1`:

- `Task001.json` (what the Studio saves)
- `Task001_products.json`

IDs must be unique within a task directory. Reference a task by its integer ID: `--task_id 1`.

---

## Required Fields

### `Description`
**Type:** `str`  
Short human‑readable explanation of the task.

---

### `Data_Path`
**Type:** `str`  
Relative path (from `data_dir`) to the dataset file.

---

### `Input_Field`
**Type:** `str`  
Column name or JSON key containing the text that should be extracted.

---

### `Parser_Format`
**Type:** `str`  
Filename of the parser module inside `tasks/parsers/` that defines a Pydantic `OutputParser` model.

`OutputParser` is the schema Extractinator validates the LLM output against.

---

## Optional Fields

### `Example_Path`
**Type:** `str`  
Relative path (from `example_dir`) to few‑shot examples.  
Required only if using `--num_examples > 0`.

---

## 4. Additional Commands

### `build-parser`
Opens the **Output Schema Builder** — a Streamlit tool for building the Pydantic `OutputParser` model visually and saving it to `tasks/parsers/`.

### `launch-extractinator`
Opens the **Studio**, the full Streamlit app for assembling datasets, output schemas, and tasks, then running them and inspecting results. See [Studio](studio.md).

