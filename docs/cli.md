# CLI Usage

The CLI is called `extractinate`.

Basic run:

```bash
extractinate --task_id 1 --model_name "phi4"
```

## Required / common options

- `--task_id <int>`  
  ID part of the task file. For `Task001_products.json`, the ID is `1`.

- `--task_dir <path>`  
  Directory that contains your task JSON files. Defaults to `tasks/`.

- `--data_dir <path>`  
  Directory that contains your input CSV/JSON. Defaults to `data/`.

- `--output_dir <path>`  
  Where to write the extracted results. Defaults to `output/`.

- `--model_name <name>`  
  Name of the Ollama model to use, e.g. `"phi4"`, `"llama3"`, …

- `--run_name <string>`  
  Optional label for this run, useful in logs.

Example with all dirs:

```bash
extractinate \
  --task_id 1 \
  --task_dir tasks/ \
  --data_dir data/ \
  --output_dir output/ \
  --model_name "phi4" \
  --run_name "baseline"
```

## Reasoning models

If you use a model that emits intermediate reasoning (e.g. DeepSeek R1), you can tell the tool to try to extract the final JSON:

```bash
extractinate --task_id 1 --model_name "deepseek-r1" --reasoning_model
```

## Output

Typical outputs:

- JSON/CSV with extracted fields
- logs for each row/document
- errors if the model could not be parsed into the schema

Always spot-check the output.
