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

## Parsers and data

Once your data is ready, you must **also** define a parser (the output structure). See the **Parser** page for that.
