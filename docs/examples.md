# Few-shot prompting

Examples are optional, but they're one of the best ways to steer the model toward the output style you want. This page explains exactly how they work and how to set them up.

---

## What an examples file is

An examples file is a **separate CSV or JSON file** containing a handful of **input → output pairs** that are shown to the model *before* it sees your real data. They shape the prompt only — they are **not** training data and are **not** used to score results.

When you run with `--num_examples > 0`, Extractinator:

1. reads the task JSON and finds `Example_Path`,
2. loads that file,
3. selects up to `num_examples` pairs by **semantic similarity** to the current input, and
4. injects them into the prompt ahead of the real case.

The real case is still validated against your `OutputParser` schema as usual.

---

## Required format

The file **must** have exactly two fields: **`input`** and **`output`** — not your dataset's `Input_Field` name, not `text`/`label`, always `input` and `output`. Both values are strings.

**JSON:**

```json
[
  {
    "input": "A 250 ml bottle of olive oil for €4.99.",
    "output": "{\"product_name\": \"Olive oil 250ml\", \"price\": 4.99}"
  },
  {
    "input": "No product mentioned in this line.",
    "output": "{\"product_name\": \"\", \"price\": 0.0}"
  }
]
```

**CSV** (quote the fields, and double any inner quotes so the JSON survives):

```text
input,output
"A 250 ml bottle of olive oil for €4.99.","{""product_name"": ""Olive oil 250ml"", ""price"": 4.99}"
```

The `output` should match the structure your schema expects. It's shown to the model as text, so nicely formatted JSON is fine.

---

## Wiring examples into a task

Add `Example_Path` (relative to `--example_dir`, default `examples/`) to your task JSON:

```json
{
  "Description": "Extract product data",
  "Data_Path": "products.csv",
  "Input_Field": "text",
  "Parser_Format": "product_schema.py",
  "Example_Path": "product_examples.json"
}
```

With `--example_dir examples/`, that file must live at `examples/product_examples.json`.

Then turn examples on at run time:

```bash
extractinate --task_id 1 --model_name "phi4" --num_examples 3
```

- `--num_examples 0` (the default) **ignores** examples even if `Example_Path` is set.
- `--num_examples N` loads up to `N` of the most relevant pairs per input.

!!! note "If you're not using examples"
    Omit `Example_Path` entirely — don't set it to an empty string.

### The embedding model

Similarity selection uses an embedding model, `nomic-embed-text` by default. Change it with `--embedding_model` (e.g. `mxbai-embed-large`). It's only used when `--num_examples > 0`.

---

## Shortcut: inline a shot or two

For just **one or two** examples, it's often simpler to bake them straight into the task `Description` rather than maintain a separate file:

```json
{
  "Description": "Extract product name and price.\n\nExample:\nInput: 'A 250ml bottle of olive oil for €4.99.'\nOutput: {\"product_name\": \"Olive oil 250ml\", \"price\": 4.99}",
  "Data_Path": "products.csv",
  "Input_Field": "text",
  "Parser_Format": "product_schema.py"
}
```

Reach for a real examples file when you want **3+ examples**, want to **reuse** them across tasks, or want to tweak them without touching the description.

---

## Troubleshooting checklist

If examples don't seem to take effect, check that:

1. `--num_examples` is **greater than 0**,
2. the task JSON has a valid `Example_Path`,
3. the file exists under `--example_dir`,
4. it has **exactly** `input` and `output` fields, and
5. those values are strings.

If all five hold, the examples are reaching the prompt — from there it's down to prompt quality and the model.
