# 💻 CLI Usage: `extractinate`

`extractinate` is a command-line tool for running and evaluating task-based generation experiments using customizable models, input data, and output settings.

## 🧰 Basic Usage

```bash
extractinate --task_id 1 --run_name "baseline_run"
```

## 🚩 Required Flags

| Flag         | Type  | Description                         |
|--------------|-------|-------------------------------------|
| `--task_id`  | `int` | ID of the task to run (required)    |

---

## ⚙️ General Options

| Flag          | Type   | Default   | Description                                 |
|---------------|--------|-----------|---------------------------------------------|
| `--run_name`  | `str`  | `"run"`   | Unique name for this run (used in logging)  |
| `--n_runs`    | `int`  | `1`       | Number of repetitions for the task run      |
| `--verbose`   | `bool` | `False`   | If set, prints detailed logs to console     |
| `--overwrite` | `bool` | `False`   | Overwrite existing outputs if they exist    |
| `--seed`      | `int`  | `None`    | Random seed for reproducibility (if set)    |

---

## 🧠 Model Configuration

| Flag               | Type    | Default   | Description                                                    |
|--------------------|---------|-----------|----------------------------------------------------------------|
| `--model_name`      | `str`   | `"phi4"`  | Name of the model to use                                       |
| `--temperature`     | `float` | `0.0`     | Sampling temperature (0 = deterministic output)                |
| `--top_k`           | `int`   | `None`    | Top-K sampling: only consider the top K predictions            |
| `--top_p`           | `float` | `None`    | Top-P sampling (nucleus): consider top tokens summing to p     |
| `--num_predict`     | `int`   | `512`     | Maximum number of tokens to generate                           |
| `--max_context_len` | `str or int`   | `"max"`   | Context length policy (`"max"` for maximum in dataset, `"split"` for splitting dataset in two parts based on length, integer number for set length)            |
| `--reasoning_model` | `bool`  | `False`   | Use for a model that performs automatic reasoning such as Deepseek-r1 (turns of JSON mode and searches the result for a valid JSON output) |

---

## 🧪 Task Input and Data Control

| Flag            | Type  | Default | Description                                                   |
|-----------------|-------|---------|---------------------------------------------------------------|
| `--num_examples`| `int` | `0`     | Number of input examples to use per task                      |
| `--chunk_size`  | `int` | `None`  | Size of input chunks (if `None`, uses full input)             |
| `--translate`   | `bool`| `False` | Whether to translate input cases to English before processing (not recommended)             |

---

## 📁 Paths and Directories

| Flag              | Type  | Default          | Description                          |
|-------------------|-------|------------------|--------------------------------------|
| `--output_dir`     | `Path`| `output/`        | Directory where results are saved    |
| `--log_dir`        | `Path`| `output/`        | Directory for logs                   |
| `--data_dir`       | `Path`| `data/`          | Directory for input data             |
| `--task_dir`       | `Path`| `tasks/`         | Directory with task configuration    |
| `--example_dir`    | `Path`| `examples/`      | Directory with example files         |
| `--translation_dir`| `Path`| `translations/`  | Directory for translated examples    |

---

## 💡 Examples

Run a task with default settings:

```bash
extractinate --task_id 1
```

Run with Llama-3.3 model and verbose logging:

```bash
extractinate --task_id 1 --model_name "llama3.3" --verbose
```

Run with a reasoning model and different temperature and number of prediction tokens:

```bash
extractinate --task_id 1 --model_name "deepseek-r1:8b" --reasoning_model --run_name "reasoning_run" --temperature 0.5 --num_predict 2048
```

---

## 📝 Notes

- Flags like `--top_k` and `--top_p` are optional but useful for controlling model creativity.
- `--num_examples` and `--chunk_size` can help manage memory and execution time on large tasks.
- If `--seed` is set, runs will be deterministic (assuming the model allows it).
