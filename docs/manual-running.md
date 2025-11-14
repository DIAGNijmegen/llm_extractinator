# Manual Running

Sometimes you want to run extraction outside the Studio and outside the CLI helper — for example, from your own Python script or a notebook.

## 1. Python entry point

```python
from llm_extractinator import extractinate

extractinate(
    task_id=1,
    model_name="phi4",
    task_dir="tasks/",
    data_dir="data/",
    output_dir="output/",
)
```

Make sure:

- the task JSON exists and is named with that ID
- the parser file referenced in the task exists
- Ollama is running and the model is available

## 2. Customizing model / client

If the package exposes a lower-level client (e.g. to override temperature or prompt), you can import that instead and pass your own settings. Keep these in your own module so you don’t have to re-edit task files.

## 3. Scheduling / automation

Because the CLI is just a thin layer, you can call it from cron, Airflow, or any workflow runner. Just make sure the environment has access to:

- the data folder
- the tasks folder
- the Ollama service
