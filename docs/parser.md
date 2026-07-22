# Output schema

The **output schema** defines what the model gives back: the fields, their types, and how they nest. Under the hood it's a **Pydantic v2 model**, and in task files it's referred to as the `Parser_Format`.

!!! info "One naming rule"
    The **top-level class must be named `OutputParser`**. That's the class Extractinator validates every model response against. Nested/helper models can be named anything.

You can create a schema two ways:

1. the **visual builder** (recommended) — no Python required, or
2. writing the Pydantic model **by hand**.

---

## 1. The visual builder

Open the standalone builder:

```bash
build-parser
```

…or use it inside the Studio: on the **Task** tab, choose *Build a new task* and click **🛠️ Build new** next to *Output schema*. Either way you get the **Output Schema Builder**, where you can:

- add fields with primitive types (`str`, `int`, `float`, `bool`), collections (`list`, `dict`), `Literal` choices, or nested models,
- mark fields optional,
- rename the model,
- preview the generated Python live, and
- **import** an existing schema file to keep editing it.

When you save, the file is written to `tasks/parsers/<name>.py`. In the Studio it's also selected for your task automatically.

---

## 2. Writing one by hand

A schema file is just a Pydantic model:

```python
from pydantic import BaseModel
from typing import Optional

class OutputParser(BaseModel):
    patient_id: str
    findings: Optional[str] = None
    measurements: Optional[list[float]] = None
```

Save it under `tasks/parsers/` and reference the filename in your task JSON.

### Nested models

Use a helper model for repeated structure, and reference it from `OutputParser`:

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float

class OutputParser(BaseModel):
    products: list[Product]
```

### Optional fields and choices

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal

class OutputParser(BaseModel):
    summary: str = Field(description="One-sentence summary of the report")
    severity: Optional[Literal["low", "medium", "high"]] = None
    followups: list[str] = []
```

`Field(description=...)` is passed to the model as guidance, so descriptive text here can improve extraction.

!!! tip "Let an LLM draft it"
    Writing schemas from scratch? Describe your fields to your favourite chat model and ask for a Pydantic v2 model with a top-level `OutputParser` class — then paste it in or import it into the builder.

---

## 3. Using the schema in a task

Reference the **filename** (not a path) in your task JSON:

```json
{
  "Parser_Format": "report.py"
}
```

Extractinator loads the model from `tasks/parsers/report.py` and instructs the LLM to return JSON matching it. When a response can't be coerced into the schema, that record is marked `"status": "failure"` with default values filled in (see [Understanding output](output.md)).

---

## Good practices

- Keep field names **lowercase and descriptive** — they double as hints to the model.
- Prefer **`Optional[...]`** for anything that might genuinely be absent, rather than forcing a value.
- **Start small.** Get two or three fields returning reliably before expanding the schema.
- Use **`Literal`** when a field should be one of a fixed set of values — it constrains the model and makes downstream code simpler.
