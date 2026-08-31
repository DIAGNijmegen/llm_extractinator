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
| `Extra_Instructions` | — | Extra text appended to the built-in system prompt (house style, output language, a domain convention) |

`Data_Path` and `Parser_Format` are **filenames**, resolved against the data and `tasks/parsers/` directories respectively — not full paths.

### What the model is told

The system prompt is assembled from three things: a fixed frame, your
`Description`, and a field guide generated from the schema — each field's name,
its type, its allowed values if it has a fixed set, whether it is optional, and
the `description` you gave it. Those descriptions are worth writing: the schema
reaches Ollama as a grammar, which enforces the *shape* of the output and
discards the prose, so the prompt is the only route by which the model learns
what a field means.

`Extra_Instructions` is appended after the built-in rules. Use it for guidance
that applies to the whole task rather than to one field — an output language, a
unit convention, a house style.

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
