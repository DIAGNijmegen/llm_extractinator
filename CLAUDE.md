# llm_extractinator

Structured extraction from unstructured clinical text using local LLMs via Ollama.
Python 3.10+. Entry points: `extractinate` (CLI), `launch-extractinator` (Streamlit Studio).

## The one invariant

```
prompt_tokens + num_predict <= num_ctx <= min(hardware_ceiling, model_native_max)
```

`num_ctx` is a **single budget the prompt and the generated answer share**, and
`num_predict` is a slice of it — not a separate allowance. On a thinking model
`num_predict` also covers the chain of thought, which is spent **first**; a
budget consumed inside `<think>` yields an empty completion and a failed row.

**Every context bug this project has had came from computing those two numbers
in more than one place.** `llm_extractinator/budget.py` is the only module
allowed to decide them. It is pure — numbers in, numbers out — so the invariant
is testable in isolation. If you find yourself adjusting `num_predict` or
`num_ctx` anywhere else, that is the bug, not the fix.

Read `.claude/rules/context-budget.md` before changing anything in the sizing
path. It loads automatically when you open those files.

## Non-obvious rules

These each cost us a release. None are apparent from reading the code alone.

- **`PredictionTask.REQUIRED_PARAMS` is a silent-drop allowlist.** A key not in
  that set is discarded with no error. This is how `max_context_cap` once never
  reached the model. Add the field there whenever you add one to `TaskConfig`.
- **Never retry a parse failure with an identical request.** Default temperature
  for a non-thinking model is 0.0, so three attempts buy three identical
  failures at three times the cost. A retry is only legitimate when something
  about the request *changes* (a larger budget, a different `think` setting) or
  when the failure was in transport. See `RETRYABLE_ERRORS` in `predictor.py`.
- **Never invent a value for a failed row.** Every schema field is `None`;
  diagnostics go in `status` / `error_type` / `error_message` / `raw_output`.
  Type defaults (`0`, `""`) are indistinguishable from real answers downstream.
- **Ollama truncates silently** when the prompt exceeds `num_ctx`
  (ollama/ollama#14259). It does not error. This is why the run refuses rather
  than proceeds when the calibrated prompt will not fit.
- **`format=` is grammar-constrained decoding, not prompt text.** The JSON
  schema does not reach the model as words — field descriptions must be rendered
  into the system prompt separately by `prompt_utils.describe_fields()`.
- **`None` means "auto"** for `--temperature`, `--num_predict` and the reasoning
  flags. Follow that convention for any new setting that has a sensible derived
  default. Do not use a magic number as both a default and a sentinel.

## Running things

```bash
pytest                      # 255 hermetic offline tests, ~7s. No Ollama needed.
pytest tests/test_budget.py -q
python devtests/run.py --model qwen3.8 --limit 3    # needs a real GPU + Ollama
```

`pytest` is the gate. `devtests/` answers only questions a faked model
structurally cannot — never put there what the offline suite could verify.
`docs/development.md` is the human entry point for both.

## Where the reasoning lives

Design decisions are recorded in prose, in two places, and both are load-bearing:

1. **Module and function docstrings.** They state *why*, not what. `budget.py`'s
   module docstring is the canonical statement of the invariant. Do not strip
   them for brevity.
2. **Project memory** (`project_memory_read`) — the accumulated notes on the
   context budget, token counting, concurrency, pipeline robustness and model
   sizing. Read `output_budget_gap.md` and `context_budget_design.md` before any
   work on sizing.

## Current work

`PLAN_0.8.0.md` is the work breakdown for this branch: tickets, lanes,
dependencies and acceptance criteria. Take work from there, not from ad-hoc
requests. Lane B and C tickets are independent and safe to run in parallel;
Lane A is sequential and must not be split.

## Conventions

- Docstrings explain the decision, not the mechanics.
- Every behavioural change gets a test in `tests/` and a `CHANGELOG.md` entry.
- Public surface is not removed, only deprecated (see
  `DataLoader.get_max_input_tokens`).
- Commit messages describe the behaviour change, not the files touched.
