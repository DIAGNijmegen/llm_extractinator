# Python

Sometimes you want to run extraction outside the Studio and outside the CLI helper — for example, from your own Python script or a notebook.
You can do this by importing the `extractinate` function from the package:

```python
from llm_extractinator import extractinate

extractinate(
    task_id=1,
    model_name="phi4",
    task_dir="/path/to/your/tasks/",
    data_dir="/path/to/your/data/",
    output_dir="/path/to/your/output/",
)
```

Make sure:

- the task JSON exists and is named with that ID
- the parser file referenced in the task exists
- Ollama is running and the model is available
