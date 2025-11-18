# Using Examples (Few‑Shot Prompting)

This page explains **how examples actually work in LLM Extractinator**, based on the current code and docs, and how to configure them correctly.

Examples are optional, but they can help steer the model toward the output style you want.

---

## 1. What examples are (and are not)

An *examples file* is a **separate CSV or JSON file** that contains a few **input → output pairs** that you want to *show* to the model before it sees the real data.

From the docs:

- You create a separate CSV/JSON file
- It has two columns/keys: **`input`** and **`output`**
- These examples are only used to build the **prompt**, they are **not** used as labels for evaluation or training

Internally, when you run with `--num_examples > 0`, Extractinator:

1. Reads the task JSON
2. Looks for `Example_Path`
3. Loads that file (CSV/JSON)
4. Takes up to `num_examples` rows/objects (decided based on semantic similarity to the current input)
5. For each example, inserts something like:

   ```text
   Example input:
   <input>

   Example output:
   <output>
   ```

   into the prompt *before* the current case.

The actual output for the current case is still parsed against the **Pydantic `OutputParser`** from your `Parser_Format` file.

---

## 2. The required key names in the examples file

Yes, the examples file **must** use specific key/column names:

> **`input`** and **`output`**

That means:

- **NOT** the same column as your main dataset’s `Input_Field`
- **NOT** arbitrary names like `text` / `label` / `expected`
- Always:

  ```csv
  input,output
  "some input text", "some target output text"
  ```

Or in JSON:

```json
[
  {
    "input": "some input text",
    "output": "some target output text"
  }
]
```

The code expects these keys and will fail if they are missing or named differently.

---

## 3. How this relates to your main dataset

Your **main dataset** (in `data/`) uses the column/key from `Input_Field` in the task JSON, e.g.:

```json
{
  "Data_Path": "reports.csv",
  "Input_Field": "text"
}
```

Your **examples file** is independent and always uses:

- `input` → a string containing example input text  
- `output` → what you want the model to produce for that input

You can either hand-craft these examples or copy realistic snippets from your main data.

---

## 4. Wiring examples into a task

In your task JSON (`tasks/TaskXXX_*.json`), you reference the examples file via `Example_Path`:

```json
{
  "Description": "Extract product data",
  "Data_Path": "products.csv",
  "Input_Field": "text",
  "Parser_Format": "product_parser.py",
  "Example_Path": "products_examples.csv"
}
```

Key points:

- `Example_Path` is **relative** to the examples directory (`--example_dir`, often `examples/`).
- Only needed if you actually want to use examples (i.e. run with `--num_examples > 0`).
- If you don’t use examples, omit `Example_Path` entirely (don’t set it to an empty string).

So with:

- `--example_dir examples/`
- `"Example_Path": "products_examples.csv"`

…the file must live at:

```text
examples/products_examples.csv
```

---

## 5. CLI flags that control examples

### `--num_examples`

```bash
extractinate --task_id 1 --model_name "phi4" --num_examples 5
```

- **Default:** `0`
- If `0`: examples are **ignored**, even if `Example_Path` exists.
- If `> 0`: Extractinator loads up to that many rows from the examples file and injects them into the prompt.

### `--example_dir`

```bash
extractinate --task_id 1 --num_examples 5 --example_dir examples/
```

- Base directory for example files.
- Combined with `Example_Path` in the task JSON.

---

## 6. What to put in `input` and `output`

### `input`

- A **short, realistic snippet** of the same type of text as in your real dataset.
- It does *not* have to be identical to any real row, but often you copy from the main data.

### `output`

- Whatever you want the model to emulate.
- In many cases, this is:

      - A JSON fragment that matches your `OutputParser` schema, or
      - A well‑structured text explanation that implicitly encodes what you want.

Example (for a parser that extracts `product_name` and `price`):

```csv
input,output
"This is a 250ml bottle of olive oil for €4.99.",
"{"product_name": "Olive oil 250ml", "price": 4.99}"
```

You can also make `output` nicely formatted JSON with newlines — it will still be shown as plain text in the prompt.

---

## 7. 1–2 shot: just inline it in the task description

For **very small numbers of examples** (1–2), it’s often **simpler to just write them directly in the task description** rather than maintaining a separate examples file.

For example, in your task JSON:

```json
{
  "Description": "Extract product name and price from each report.\n\nExample:\nInput: 'This is a 250ml bottle of olive oil for €4.99.'\nOutput: {"product_name": "Olive oil 250ml", "price": 4.99}",
  "Data_Path": "products.csv",
  "Input_Field": "text",
  "Parser_Format": "product_parser.py"
}
```

This gives you a **one-shot example** baked into the system prompt without having to deal with:

- `Example_Path`
- `--num_examples`
- A separate examples CSV/JSON

Use a dedicated examples file when:

- You want **3+ examples**
- You want to re‑use the same examples across multiple tasks
- You want to tweak examples without touching the task description

---

## 8. Troubleshooting checklist

If examples don’t seem to be used:

1. `--num_examples` is **> 0**.
2. The task JSON has a valid `Example_Path`.
3. The file exists under `--example_dir`.
4. The examples file has **exactly** `input` and `output` keys / columns.
5. The values are strings (especially for `input`).

If all of the above are true, the examples are being injected into the prompt; from there, it’s about prompt quality and model behavior.

---

## 9. Summary

- Examples live in a *separate* CSV/JSON, referenced via `Example_Path`.
- The file must use **`input`** and **`output`** as keys/columns.
- They’re only used for **few‑shot prompting**, not for training.
- Use `--num_examples` to turn them on.
- For 1–2 shots, it’s often easiest to **inline** the example in the task description instead of creating an examples file.
