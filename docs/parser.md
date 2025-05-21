# 🛠️ `build-parser` – Interactive Pydantic Model Builder

The `build-parser` command launches an intuitive **web-based tool** to visually create [Pydantic v2](https://docs.pydantic.dev/latest/) models—without writing a single line of code. It's great for designing **OutputParser formats**, structured data schemas, or simply organizing typed data models.

---

## 🚀 Getting Started

Make sure your Python package is installed in an environment where the CLI is available. Then, simply run:

```bash
build-parser
```

This command launches a local Streamlit app in your browser.

---

## 📦 OutputParser Root Model

At the core of this tool is the **`OutputParser`** class. This is the primary model used by the LLM in your framework to parse outputs into structured JSON.

You always start with the `OutputParser` model, and build its fields just like any Pydantic model. You can then create **additional models** to use as nested types or components within `OutputParser`.

This approach supports building **complex, hierarchical schemas** using composition and reuse.

---

## 🧱 Interface Overview

### 🧭 Model Manager (Sidebar)

The Model Manager allows you to create additional models beyond `OutputParser`.

### Why is this useful?

Often, structured outputs require **nested objects** or repeated elements. Instead of defining everything flatly in `OutputParser`, you can:

- Break down your schema into logical subcomponents
- Reuse those components across multiple fields
- Keep your data models clean, readable, and maintainable

#### Example:

You could define a `Product` model and use it as a field in `OutputParser` like:

```python
class OutputParser(BaseModel):
    products: list[Product]
```

#### How to use it:

- Enter a model name (e.g., `Product`, `Metadata`)
- Click ➕ Add model
- Names must begin with a capital letter and be valid Python identifiers

---

### 🏗️ Design Tab

Use this section to add fields to `OutputParser` and any other models you've created.

  - Field name
  - Field type (primitive, special type, or reference to another model)
  - Mark as `Optional` if needed
  - Provide subtypes for `list`/`dict`, or values for `Literal`

---

### 📝 Code Tab

This shows the full generated code using [Pydantic v2](https://docs.pydantic.dev/latest/):

- Starts with the `OutputParser` model
- Includes any additional models you’ve defined

---

### 💾 Export Tab

Choose to either:

- **Download** the generated code as a `.py` file
- Or **save directly** to `tasks/parsers/` inside your project

---

## 🧪 Example

Let’s say you want to extract a list of products from LLM output:

1. Define a model called `Product` with fields like:
   - `name`: `str`
   - `price`: `float`
2. In `OutputParser`, add:
   - `products`: `list[Product]`

Resulting code:

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float

class OutputParser(BaseModel):
    products: list[Product]
```

---

## 📁 Output Location

If you choose **"Save to `tasks/parsers/`"**, your file will be saved at:

```bash
<project-root>/tasks/parsers/<filename>.py
```
