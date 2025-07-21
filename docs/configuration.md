# 🛠️ Task Configuration

Create a JSON task file in the `tasks/` folder using the following naming format:

```bash
TaskXXX_taskname.json
```

---

## 📌 Required Fields

Your JSON file must include the following keys:

- `Description`: Description of the task  
- `Data_Path`: Filename of the dataset  
- `Input_Field`: Column containing the text data  
- `Parser_Format`: The filename of the file containing the parser format in the `tasks/parsers/` folder

---

### 🔍 About `Parser_Format`

This field should reference a Python file located in `tasks/parsers/` which defines a Pydantic model called `OutputParser`.

This file is usually created using the `build-parser` tool. It must contain a class named `OutputParser`, which will be used by the LLM to structure its response into a valid JSON object.

You may define other nested models to keep the schema organized, but `OutputParser` is the root that the system will rely on.

#### Example

```json
"Parser_Format": "product_parser.py"
```

The file `tasks/parsers/product_parser.py` might look like:

```python
from pydantic import BaseModel

class Product(BaseModel):
    name: str
    price: float

class OutputParser(BaseModel):
    products: list[Product]
```

---

## 🔄 Optional (for using examples)

- `Example_Path`: Path to the file with examples to include in the prompt

> **Note:** If you're not using examples, simply omit the `Example_Path` field.  
> Do **not** set it to an empty string!
