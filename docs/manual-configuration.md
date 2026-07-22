# Manual configuration

Prefer to work in files rather than the Studio? Everything the Studio produces is plain text you can create by hand. This page covers the layout and conventions.

## Directory layout

A typical project:

```text
.
├── data/
│   └── reports.csv              # your source data
├── tasks/
│   ├── Task001_reports.json     # a task file
│   └── parsers/
│       └── report.py            # the output schema (Pydantic model)
├── examples/                    # optional few-shot files
└── output/                      # results are written here
```

These map to the CLI's default directories (`--data_dir`, `--task_dir`, `--example_dir`, `--output_dir`), so if you keep this layout you rarely need to pass any path flags.

## The task file

A minimal task:

```json
{
  "Description": "Extract structured info from radiology reports",
  "Data_Path": "reports.csv",
  "Input_Field": "text",
  "Parser_Format": "report.py"
}
```

| Field | Required | Notes |
|---|---|---|
| `Description` | ✅ | Plain-language instruction to the model |
| `Data_Path` | ✅ | Dataset filename, relative to `--data_dir` |
| `Input_Field` | ✅ | Exact column/key holding the text |
| `Parser_Format` | ✅ | Schema filename in `tasks/parsers/` |
| `Example_Path` | — | Few-shot file, relative to `--example_dir` (only if using examples) |

`Data_Path` and `Parser_Format` are **filenames**, resolved against the data and `tasks/parsers/` directories respectively — not full paths.

## Naming task files

Task files must be named so the CLI can extract the numeric ID:

```text
Task<NNN>...json
```

- `<NNN>` is a **zero-padded three-digit** number — this is the ID.
- Anything may follow it (an underscore and a descriptive name is common), as long as it isn't another digit.

Examples:

| Filename | `--task_id` |
|---|---|
| `Task001.json` | `1` |
| `Task001_reports.json` | `1` |
| `Task010_products.json` | `10` |

Each ID must be **unique** within a task directory — two files with the same ID is an error. Always reference tasks by their integer ID on the CLI (`--task_id 10`).

## The output schema

`Parser_Format` points at a Python file in `tasks/parsers/` defining a Pydantic model whose top-level class is named `OutputParser`. See [Output schema](parser.md) for how to write one (or generate it with `build-parser`).

## Next

Run your hand-built task exactly like any other:

```bash
extractinate --task_id 1 --model_name "phi4"
```
