# Manual Configuration

This page is for users who want to create task files and parsers **by hand** without the Studio.

## Directory layout

A common layout is:

```text
.
├── data/
│   └── reports.csv
├── tasks/
│   ├── Task001_reports.json
│   └── parsers/
│       └── report.py
└── output/
```

- `data/` — your source CSV/JSON
- `tasks/` — your task JSONs
- `tasks/parsers/` — Python files with Pydantic models
- `output/` — where extraction results go

## Task JSON fields

Minimum viable task:

```json
{
  "Description": "Extract structured info from radiology reports",
  "Data_Path": "data/reports.csv",
  "Input_Field": "text",
  "Parser_Format": "report.py"
}
```

The only additional field you may want is `Example_Path` if you intend to use example-based prompting.

## Naming

Use the `TaskXXX_name.json` style so the CLI can pick task IDs reliably:

- `Task001_reports.json` → `--task_id 1`
- `Task010_products.json` → `--task_id 10`

Stick to **integers** in the CLI to avoid confusion.
