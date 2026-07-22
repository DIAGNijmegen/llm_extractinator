# Python API

Beyond the Studio and the CLI, you can call extraction directly from a Python script or notebook with the `extractinate` function.

```python
from llm_extractinator import extractinate

extractinate(
    task_id=1,
    model_name="phi4",
)
```

## Passing options

`extractinate` accepts the same options as the CLI, as keyword arguments (drop the leading `--`). Paths can be strings or `Path` objects:

```python
from llm_extractinator import extractinate

extractinate(
    task_id=1,
    model_name="phi4",
    num_examples=3,
    reasoning_model=False,
    temperature=0.0,
    task_dir="/path/to/tasks/",
    data_dir="/path/to/data/",
    output_dir="/path/to/output/",
)
```

Any option you don't pass falls back to its default (see the [settings reference](settings-reference.md)). Unlike the CLI, there's no shell to manage directories for you, so pass `task_dir` / `data_dir` / `output_dir` explicitly unless you're running from a folder with the standard layout.

## Before you call it

Make sure that:

- the task JSON exists and is named for that ID (`Task001*.json` for `task_id=1`),
- the schema file it references exists in `tasks/parsers/`, and
- Ollama is running (the model is pulled automatically on first use).

## Reading the results

`extractinate` writes results to disk (it doesn't return them). Load the output JSON afterwards:

```python
import pandas as pd

df = pd.read_json("output/run/Task001-run0/nlp-predictions-dataset.json")
print(df["status"].value_counts())
```

See [Understanding output](output.md) for the file layout and record shape.
