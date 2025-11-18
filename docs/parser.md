# Parser

The parser defines the **output shape**: the fields, their types, and nesting. Internally this is a **Pydantic** model.

You can create it in two ways:

1. using the **visual builder** (recommended): `build-parser`
2. writing the Pydantic model by hand

## 1. Visual builder

Run:

```bash
build-parser
```

This opens the UI where you can:

- add fields (string, int, float, list, nested model)
- rename the model
- preview the generated Python
- export it

When you export, save the file to:

```text
tasks/parsers/<name>.py
```

## 2. Structure of the generated file

A generated parser file usually looks like:

```python
from pydantic import BaseModel
from typing import Optional, List

class OutputParser(BaseModel):
    patient_id: str
    findings: Optional[str] = None
    measurements: Optional[List[float]] = None
```

You can edit this file manually if you want to add validators or docstrings.

!!! tip
    You can also write your own Pydantic models from scratch. Just make sure they inherit from `BaseModel` and the top level class is named `OutputParser`. You can ask your favorite LLM to help you with setting it up for your specific use case!

## 3. Using the parser in a task

In your task JSON:

```json
{
  "Parser_Format": "report.py"
}
```

The extractor will load the Python model from `tasks/parsers/report.py` and tell the LLM to return JSON that matches it.

## 4. Good practices

- keep field names lowercase and descriptive
- prefer `Optional[...]` for fields that might not be present in the text
- start with a small model and expand it once the LLM returns consistent data
