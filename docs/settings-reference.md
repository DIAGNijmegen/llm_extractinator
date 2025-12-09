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
| `--embedding_model` | `"nomic-embed-text"` | Embedding model for few‑shot selection. |
| `--temperature` | `0.0` | Sampling randomness. |
| `--top_k` | `None` | Top‑K sampling. |
| `--top_p` | `None` | Nucleus sampling. |
| `--num_predict` | `512` | Maximum generated tokens. |
| `--max_context_len` | `"max"` | Context length strategy. |
| `--reasoning_model` | `False` | Enables reasoning‑model mode. |
| `--num_examples` | `0` | Number of few‑shot examples. |
| `--chunk_size` | `None` | Chunk size for long inputs. |
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

### `--embedding_model`

**Type:** `str`
**Default:** `"nomic-embed-text"`
Name of the embedding model to use for few-shot example selection via semantic similarity. Only used when `--num_examples > 0`. See [Ollama models](https://ollama.com/models) for available embedding models (e.g., `"mxbai-embed-large"`, `"nomic-embed-text"`).

---

### `--temperature`

**Type:** `float`
**Default:** `0.0`  
Controls randomness in generation:
- `0.0` = deterministic  
- Higher values = more creative output

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
**Default:** `512`  
Maximum number of tokens to generate for the model’s output.

---

### `--max_context_len`
**Type:** `str` or `int`  
**Default:** `"max"`  
Controls context length policy:
- `"max"` — use maximum available length  
- `"split"` — split dataset in two by input size  
- integer — explicitly set context length

---

### `--reasoning_model`
**Type:** `bool`  
**Default:** `False`  
Enable this for models like DeepSeek‑R1 and Qwen3 that output chain‑of‑thought before JSON.  
Enabling this flag allows the model to emit reasoning steps prior to the final answer extraction.

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

Task files define what to extract and how to parse it.  
Files follow the pattern:

```
TaskXXX_name.json
```

Example:  
`Task001_products.json` → task ID `1`.

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
Launches a Streamlit tool for interactively building Pydantic parser models.

### `launch-extractinator`
Opens the Streamlit GUI for assembling datasets, parsers, and tasks.

