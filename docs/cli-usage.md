# 💻 CLI Usage: `extractinate`

Below are the configurable input flags for the CLI tool:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--task_id`         | `int`    | **Required**     | Task ID to run |
| `--run_name`        | `str`    | `"run"`          | Logging name |
| `--n_runs`          | `int`    | `0`              | Number of runs |
| `--num_examples`    | `int`    | `0`              | Examples per task |
| `--num_predict`     | `int`    | `512`            | Max prediction tokens |
| `--chunk_size`      | `int`    | `None`           | Chunk size |
| `--overwrite`       | `bool`   | `False`          | Overwrite existing outputs |
| `--translate`       | `bool`   | `False`          | Translate examples |
| `--verbose`         | `bool`   | `False`          | Verbose output |
| `--reasoning_model` | `bool`   | `False`          | Use reasoning model |
| `--model_name`      | `str`    | `"phi4"` | Model name |
| `--temperature`     | `float`  | `0.0`            | Generation temperature |
| `--max_context_len` | `str`    | `"max"`          | Context length behavior |
| `--top_k`           | `int`    | `None`           | Top-K sampling |
| `--top_p`           | `float`  | `None`           | Nucleus sampling |
| `--seed`            | `int`    | `None`           | Random seed |
| `--output_dir`      | `Path`   | `output/`        | Output directory |
| `--task_dir`        | `Path`   | `tasks/`         | Task config directory |
| `--log_dir`         | `Path`   | `output/`        | Log directory |
| `--data_dir`        | `Path`   | `data/`          | Input data path |
| `--example_dir`     | `Path`   | `examples/`      | Examples directory |
| `--translation_dir` | `Path`   | `translations/`  | Translation directory |
