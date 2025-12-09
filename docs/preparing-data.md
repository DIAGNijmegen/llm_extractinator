# Preparing Data

LLM Extractinator expects that each **row (CSV)** or **item (JSON)** contains the text you want to extract from.

The extractor needs to know **which field/column** to read — you’ll tell it that in the **task JSON** (`"Input_Field": "text"`).

## CSV example

```text
id,text
1,"This is the first report..."
2,"This is the second report..."
```

- `text` is the column we will extract from
- you can have more columns; they will be carried through if the task supports it

## JSON example

```json
[
  {
    "id": 1,
    "text": "This is the first report..."
  },
  {
    "id": 2,
    "text": "This is the second report..."
  }
]
```

Again, `text` is the field we will extract from.

## Data path

In your task JSON you will refer to the data:

```json
{
  "Data_Path": "reports.csv",
  "Input_Field": "text"
}
```

- `Data_Path` is relative to the data directory you pass to the CLI (`--data_dir`)
- `Input_Field` must match the column/key name exactly

## Few-shot examples

If you want to use few-shot learning (providing examples to guide the model), you can create an examples file. Set `--num_examples > 0` and reference the file in your task JSON.

### Example file format

Few-shot examples should be in **JSON format** with `input` and `output` fields:

```json
[
  {
    "input": "Patient presented with severe headache and nausea.",
    "output": "{\"symptoms\": [\"headache\", \"nausea\"], \"severity\": \"severe\"}"
  },
  {
    "input": "No significant findings. Patient is in good health.",
    "output": "{\"symptoms\": [], \"severity\": null}"
  }
]
```

**Important notes:**

- Each example must have an `input` field (the text to extract from)
- Each example must have an `output` field (the expected JSON output as a string)
- The output should match the structure defined by your parser
- Store examples in the `examples/` directory (or the path specified by `--example_dir`)

### Adding examples to your task

In your task JSON, add the `Example_Path` field:

```json
{
  "Description": "Extract medical symptoms",
  "Data_Path": "medical_reports.csv",
  "Input_Field": "text",
  "Parser_Format": "medical_parser.py",
  "Example_Path": "medical_examples.json"
}
```

Then run with `--num_examples`:

```bash
extractinate --task_id 1 --model_name "phi4" --num_examples 3
```

This will use semantic similarity to select the 3 most relevant examples for each input.

### Configuring the embedding model

By default, `nomic-embed-text` is used for finding similar examples. You can change this:

```bash
extractinate --task_id 1 --num_examples 3 --embedding_model "mxbai-embed-large"
```

---

## Parsers and data

Once your data is ready, you must **also** define a parser (the output structure). See the **Parser** page for that.
