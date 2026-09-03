# llm_extractinator

Extracts structured data from unstructured clinical text using local LLMs through
Ollama. A user describes an extraction *task* in a JSON file, points it at a
dataset of documents, and gets one validated JSON object per document. Built for
hospital use, which is why everything runs locally and why silent wrong answers
are treated as worse than loud failures.

Python 3.10+. Entry points: `extractinate` (CLI), `launch-extractinator`
(Streamlit Studio), `build-parser` (schema builder).

## Vocabulary

Use these words the way the codebase does. Getting them confused is the most
common source of a wrong change here.

| Term | Meaning |
|---|---|
| **task** | One extraction job, defined by `tasks/TaskNNN_name.json`. The number is the id passed as `--task_id`. |
| **task file** | That JSON: `Description`, `Data_Path`, `Input_Field`, `Parser_Format`, optional `Example_Path` and `Extra_Instructions`. |
| **`Parser_Format`** | The output schema. Either an inline dict of field definitions, or the filename of a Python module under `tasks/parsers/` exporting an `OutputParser` pydantic model. Both resolve through `output_parsers.resolve_parser_model`. |
| **`Input_Field`** | The column in the data file holding the document text. |
| **examples** | Labelled input/output pairs for few-shot prompting, selected per row by similarity (MMR over embeddings). `--num_examples 0` means zero-shot, and is the default. |
| **run** | One pass over the data. `--n_runs` repeats it; each gets its own output folder. |
| **test run** | `--test_run_size N` — process N rows but **size the window from the full dataset**, so a small check does not pass with a window the real run never uses. |
| **split mode** | `--max_context_len split` — partition documents into short and long, run each phase with its own right-sized window. Worth it when lengths vary a lot. |
| **chunking** | `--chunk_size N` — write predictions every N rows so a long run is resumable. Unrelated to context size. |
| **translation** | An optional pre-pass translating documents to English before extraction. |

## Directory layout

Paths default relative to the working directory and are all overridable.

```
tasks/         TaskNNN_*.json + parsers/     what to extract
data/          the documents                 Data_Path resolves here
examples/      few-shot pairs                Example_Path resolves here
translations/  cached translation output
output/
  logs/        task_runner.log, ollama_server.log
  <run_name>/<task_name>[-testN]-run<i>/
      nlp-predictions-dataset.json           one object per input row
      failures.json                          written only when rows failed
devtests/      GPU harness (not collected by pytest, not shipped)
docs/          mkdocs source
```

## How a run flows

`extractinate(**kwargs)` → `TaskConfig` → `TaskRunner.run_tasks()`:

1. `_extract_task_info()` — `TaskLoader` finds and loads the task file.
2. `_load_data()` — `DataLoader` reads documents and examples, adds a
   `token_count` column.
3. **Server up, and it stays up for the whole run** (one `try`/`finally`).
4. `pull_model()`, then `model_capabilities()` — one `/api/show` giving
   `supports_thinking` and `native_context`.
5. `resolve_thinking()` — does this run reason?
6. `_output_budget()` — the generation budget, **finalised before the window**.
7. `_run_phases()` → `_run_phase()` → `resolve_budget()` → `PredictionTask`.
8. `PredictionTask.run()` → `prepare_prompt_ollama()` →
   `_calibrate_prompt_estimate()` (measure the real prompt, grow the window if
   the estimate was short) → `Predictor.predict()`.
9. `Predictor.predict()` — `chain.batch` over rows, `format=<schema>` bound for
   grammar-constrained decoding, failures via `validator.handle_prediction_failure`.

**The ordering in 3–7 is load-bearing, not incidental.** The model must exist
before it can be asked what it supports; what it supports decides the generation
budget; the generation budget decides the window. Sizing used to happen above
step 3, before any of it was knowable.

## Module map

| File | Responsibility |
|---|---|
| `main.py` | CLI, `TaskConfig`, `TaskRunner` — orchestration and phase dispatch. |
| `budget.py` | **The only** decider of `num_ctx` and `num_predict`. Pure. |
| `data_loader.py` | `DataLoader` (reading, token counts, estimates, splitting) and `TaskLoader`. |
| `prediction_task.py` | One phase: builds the `ChatOllama`, calibrates, runs, writes output. |
| `predictor.py` | Prompt assembly, the LangChain chain, batching, outcome logging. |
| `prompt_utils.py` | Renders the system prompt: description, field guide, extra instructions. |
| `output_parsers.py` | Task schema → pydantic model; output-size estimation. |
| `validator.py` | What a failed row carries. |
| `callbacks.py` | Row-accurate progress bar. |
| `ollama_server.py` | Server lifecycle, `/api/show` capabilities, prompt measurement. |
| `run_config.py` | `RunSettings` → CLI command. Pure, so the Studio is testable. |
| `gui.py` | Streamlit Studio. Runs Streamlit at import — keep logic out of it. |
| `schema_builder.py` | Standalone UI for authoring a `Parser_Format`. |
| `translator.py`, `utils.py`, `theme.py` | Translation pre-pass, IO helpers, Studio theme. |

## The one invariant

```
prompt_tokens + num_predict <= num_ctx <= min(hardware_ceiling, model_native_max)
```

`num_ctx` is a **single budget the prompt and the generated answer share**. On a
thinking model it also covers the chain of thought, which is spent **first** — a
budget consumed inside the reasoning channel yields an empty completion and a
failed row.

Every context bug this project has had was that relationship computed in more
than one place. `budget.py` exists so there is exactly one. If you are adjusting
`num_ctx` or `num_predict` anywhere else, that is the bug, not the fix.
`.claude/rules/context-budget.md` loads automatically when you open the sizing
path; read it before changing anything there.

## Non-obvious rules

Each of these cost a release. None are apparent from the code alone.

- **`PredictionTask.REQUIRED_PARAMS` is a silent-drop allowlist.** A key not in
  that set is discarded with no error. This is how `max_context_cap` once never
  reached the model.
- **Never retry a request that is byte-identical.** Default temperature for a
  non-thinking model is 0.0, so three attempts buy three identical failures at
  three times the cost. A retry is legitimate only when the *request* changes or
  the failure was in transport. See `RETRYABLE_ERRORS`.
- **Never invent a value for a failed row.** Every schema field is `None`; the
  diagnosis goes in `status` / `error_type` / `error_message` / `raw_output`. A
  type default (`0`, `""`) is indistinguishable from a real answer downstream,
  and a fabricated `0` in a measurement column is the dangerous case.
- **Ollama truncates the prompt silently** past `num_ctx` (ollama/ollama#14259).
  It does not error. That is why a run refuses rather than proceeds when the
  calibrated prompt will not fit.
- **`format=` is grammar-constrained decoding, not prompt text.** The JSON schema
  never reaches the model as words. Field descriptions must be rendered into the
  system prompt separately by `prompt_utils.describe_fields()`.
- **`None` means "auto"** for `--temperature`, `--num_predict` and the reasoning
  flags. Never use a magic number as both a default and a sentinel — `num_predict`
  was once `512` meaning three different things at once.
- **`gui.py` runs Streamlit at import time**, so nothing in it can be unit
  tested. That is why `run_config.py` exists and why logic belongs there.

## Adding a setting

A new flag has to land in eight places. Missing any of them fails silently,
which is the whole reason this checklist exists.

1. `main.py` — `parse_args()` argument.
2. `main.py` — `TaskConfig` field, with `None` as auto where that applies.
3. `main.py` — `TaskConfig.__post_init__` validation, if the value can be wrong.
4. `prediction_task.py` — **`REQUIRED_PARAMS`**. Nothing catches this omission.
5. `run_config.py` — `RunSettings` field and `to_command()` (emit only when it
   differs from the default, so the shown command stays readable).
6. `gui.py` — widget, plus a `_WIDGET_DEFAULTS` entry (every widget seeds through
   session state; a `value=` alongside a `key=` is silently ignored).
7. `docs/cli.md` and `docs/settings-reference.md`.
8. `tests/test_cli.py` and `tests/test_run_config.py`.

## Commands

```bash
pytest                      # 277 hermetic tests. No Ollama, no network, no GPU.
pytest -q --no-cov          # use this when running agents in parallel: addopts
                            # sets --cov, and concurrent runs fight over .coverage
pytest tests/test_budget.py -q

python devtests/run.py --model qwen3.8 --limit 3     # needs a real GPU + Ollama
python devtests/run.py --list

mkdocs serve                # docs preview
```

`pytest` is the gate. Wall time varies a lot by platform (seconds on Linux,
minutes on Windows under `--cov`), so judge it against *this machine's* previous
run rather than an absolute number. A suite that suddenly got much slower is a
finding: the usual cause is something reaching the network, most often a tiktoken
vocabulary fetch. `pytest -q --no-cov --durations=20` tells the two apart — time
spread evenly across tests is instrumentation, time concentrated in a few
token-counting tests is the network.

`devtests/` answers only what a faked model **structurally cannot** — never put
there what the offline suite could verify. `docs/development.md` is the human
entry point for both.

## Release process

- Version lives in `llm_extractinator/__init__.py`, nowhere else
  (`pyproject.toml` reads it dynamically).
- Work happens on a branch named for the version (`0.7.0`, `0.8.0`).
- `CHANGELOG.md` gets an entry per behavioural change, written from the user's
  point of view.
- Merging to `main` **publishes to PyPI** via `.github/workflows/publish.yml`.
  Nothing else guards that, so `main` is a release, not a staging area.

## Conventions

- **Docstrings explain the decision, not the mechanics.** This project records
  its reasoning in prose, and that prose is load-bearing: a change that
  contradicts a docstring is wrong even when the tests pass. If the docstring is
  wrong, say so — do not silently overrule it.
- Every behavioural change gets a test.
- Public surface is deprecated, not removed (see `DataLoader.get_max_input_tokens`).
- Commit messages describe the behaviour change, not the files touched.

## Where the rest of the reasoning lives

1. **Module docstrings** — `budget.py`'s is the canonical statement of the
   invariant. Do not strip them for brevity.
2. **Project memory** (`project_memory_read`) — accumulated notes on the context
   budget, token counting, concurrency, pipeline robustness, model sizing and the
   agentic setup. Read `output_budget_gap.md` and `context_budget_design.md`
   before any work on sizing.
3. **`.claude/rules/`** — path-scoped detail that loads only when you open the
   files it covers.

## Current work

`PLAN_0.8.0.md` is the work breakdown for this branch: tickets, lanes, file
scopes, acceptance criteria and execution waves. Take work from there rather than
from ad-hoc requests. Lane B and C tickets are independent and safe to delegate;
Lane A is sequential and must not be split. Delete the plan when 0.8.0 ships.
