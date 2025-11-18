# CLI Usage

The CLI is called `extractinate`.

Basic run:

```bash
extractinate --task_id 1 --model_name "phi4"
```

## Required options

- `--task_id <int>`  
  ID part of the task file. For `Task001_products.json`, the ID is `1`.

For a complete list of options, check out [Settings Reference](settings-reference.md).

## Reasoning models

If you use a model that emits intermediate reasoning (e.g. DeepSeek R1, Qwen3, etc.), you can tell the tool to try to extract the final JSON:

```bash
extractinate --task_id 1 --model_name "deepseek-r1" --reasoning_model
```

## Output

Typical outputs:

- JSON/CSV with extracted fields
- logs for each row/document
- errors if the model could not be parsed into the schema

Always spot-check the output.
