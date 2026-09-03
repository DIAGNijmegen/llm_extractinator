# 0.8.0 — bounding the chain of thought

**Branch:** `0.8.0` (not `0.7.1`: this adds a CLI flag and a pipeline phase, which
is a minor bump under the convention every previous branch here has followed.)

## The problem, settled

Confirmed on a real run — qwen3.8, Dutch RECIST reports, 67 failures in ~855 rows
(7.8%), every one identical:

```
error_type:    OutputParserException   67/67
error_message: "Invalid json output: \n"  67/67
raw_output:    ""                      67/67
```

The model returns **zero content**. The chain of thought consumes the whole
generation budget and the JSON is never reached. `num_predict` covers thinking
and answer together, and thinking is spent first.

This is not a defect in this package. [ollama/ollama#17561](https://github.com/ollama/ollama/issues/17561)
is an open proposal to bound thinking separately; today there is no
`think_budget`, and the issue names this exact symptom on Qwen-family models.
The only levers that exist are **shortening the trace** (`think: "low"`) and
**not thinking** (`think: false`).

Two further facts from the same run:

- `raw_output` is empty because it captures the *content* channel only. With
  `reasoning=True` the trace goes to `additional_kwargs["reasoning_content"]`
  and is discarded before the exception is raised — so the one artifact that
  would say whether the model was looping or merely slow is thrown away.
- The budget log is misleading. `context 33441 = 8441 prompt + 25000 answer + 0
  headroom (fitted to the data)` described a window three quarters of which came
  from a user-set `--num_predict`. `Budget.source` reports "fitted to the data"
  whenever `requested_ctx is None`, regardless of what actually filled it.

## What ships in 0.8.0

Prevention, then recovery, then honesty about both.

1. **`--reasoning_effort`** — expose Ollama's thinking levels. Shortens the trace
   at the source and cuts wall-clock at the same time.
2. **Metadata capture** — `done_reason`, `eval_count` and the reasoning channel,
   so truncation is diagnosable and the budget is sizable from measurement.
3. **A retry pass** — one second pass over failed rows only, at a larger window
   and reduced thinking, marked as degraded.
4. **Naming fixes** — `Budget.source`, and the three confusable token fields in
   the Studio.
5. **The loose ends** — five small independent defects already identified.

Explicitly **not** in 0.8.0, with reasons, in `Deferred` at the bottom.

---

## Lane A — the reasoning budget

**Sequential. One owner. Not delegated to parallel agents.**

The reason is specific rather than precautionary: A1 changes how the reasoning
state is resolved, A2 changes what a failure carries, and A3 consumes both. Three
agents each holding one third of that would reproduce, organisationally, the
exact bug class this codebase exists to have fixed — one rule computed in several
places by parties who each know part of it. Run these in order, in one session,
and dispatch `budget-guardian` after A3.

### A1 — `--reasoning_effort {low,medium,high,max}`

**Scope:** `main.py` (flag + `TaskConfig`), `run_config.py`, `prediction_task.py`,
`gui.py`, `tests/test_run_config.py`, `tests/test_cli.py`, `tests/test_prediction_task.py`

Ollama accepts `think: "low"|"medium"|"high"|"max"` on most reasoning models.
`ChatOllama`'s `reasoning=` parameter takes those strings. gpt-oss accepts **only**
levels and silently ignores booleans, so the current True/False/None resolution is
already incomplete for that family.

Design: keep `resolve_thinking()` returning `bool` — the budget only needs to know
*whether* this run reasons — and add a separate resolver for the wire value:

```python
resolve_reasoning_param(detected, reasoning_model, no_reasoning, effort)
    -> True | False | None | "low" | "medium" | "high" | "max"
```

That preserves the property that the runner and the task cannot disagree, while
keeping the four-state wire value out of the budget's way. Precedence:
`no_reasoning` beats everything; an explicit `effort` implies reasoning is on.

Follow the `None = auto` convention. Do **not** make `"medium"` the default —
auto stays auto.

**Acceptance**
- `--reasoning_effort low` reaches `ChatOllama(reasoning="low")`, pinned by a test.
- `--no_reasoning --reasoning_effort high` resolves to reasoning off, with a warning.
- The flag round-trips through `RunSettings.to_command()`.
- The field is in `PredictionTask.REQUIRED_PARAMS`.
- `pytest -q` green.

### A2 — capture what the model actually did

**Scope:** `predictor.py`, `validator.py`, `tests/test_failure_reporting.py`

Today the chain is `bound_llm | _strip_think_tags | base_parser`. The parser
consumes the `AIMessage`, so `response_metadata` is dropped on success and only
`llm_output` survives on failure.

Design: classify inside the chain rather than correlating callbacks. Insert a step
**after** `_strip_think_tags` (which preserves `response_metadata` through
`model_copy`) and **before** the parser, which raises a typed exception carrying
the metadata:

```
bound_llm | _strip_think_tags | _classify_completion | base_parser
```

- content empty after stripping → `EmptyCompletion`
- `done_reason == "length"` → `TruncatedOutput`
- otherwise pass through

The exception carries the metadata itself, so no run-id correlation is needed —
which is the trap here, since `chain.batch` runs rows concurrently and the parse
failure has no handle on the callback's `parent_run_id`.

New diagnostic columns on every row: `done_reason`, `eval_count`, and a clipped
tail of `reasoning_content`. The reasoning tail is the point — it is what
distinguishes a long-but-finite trace (retry will work) from a repetition loop
(retry will not).

For the distribution, an ordered list of `eval_count` is enough; per-row
correlation on *successful* rows is not required and is not worth restructuring
the chain for.

**Acceptance**
- A faked response with `done_reason: "length"` produces `error_type:
  "TruncatedOutput"`, not `OutputParserException`.
- A faked response with empty content produces `EmptyCompletion`.
- Failure rows carry a non-empty reasoning tail when the fake supplies one.
- End of run logs p50 / p95 / max `eval_count` against `num_predict`.
- Successful rows still carry the full diagnostic key set as `None`.
- `pytest -q` green.

### A3 — the retry pass

**Scope:** `prediction_task.py`, `main.py`, `utils.py`, `tests/test_prediction_task.py`,
`tests/test_pipeline_offline.py`

A **second pass over failed rows only**, not a per-row retry. Changing `num_ctx`
reallocates the KV cache, which means a model reload; per row that is ruinous,
once per pass it is cheap and it fits the existing phase structure.

```
main pass    num_predict = N        think = resolved
retry pass   num_predict = 2N       think = one level lower (or False)
```

Escalating both is deliberate. Budget alone loses to a repetition loop; reduced
thinking alone loses to a genuinely long trace. Together they cover both, and the
retry is a materially different request — which is what makes it legitimate under
the project's no-identical-retry rule.

Constraints:
- **Eligibility:** only `EmptyCompletion` and `TruncatedOutput` rows. A transport
  failure or a genuine schema violation is not retried here.
- **One pass. No ladder, no loop.**
- **Clamped:** the escalated window goes through `resolve_budget` under the same
  `max_context_cap` and `model_native_max`. If it cannot grow, skip the pass and
  say so rather than reloading for nothing.
- **VRAM:** Ollama multiplies `num_ctx` by `OLLAMA_NUM_PARALLEL`. A doubled window
  on a server with four slots is eight times the KV cache. Log the escalated
  window prominently; full slot-awareness is deferred (see below).
- **No eligible failures → no reload.** The common case must cost nothing.
- **Marking:** recovered rows carry `attempts: 2` and `degraded: true`. They were
  answered with less reasoning and are not equivalent to a first-pass row.
  Consistent with the existing rule that a row never hides how it was produced.
- Results merge back by row index. Must be correct under `chunk_size`, `n_runs`
  and `split`.

**Acceptance**
- A run with zero eligible failures performs no reload and no second pass.
- A run with eligible failures re-runs exactly those rows and merges by index.
- Recovered rows carry `attempts` and `degraded`; first-pass rows do not.
- A row that fails twice keeps its **second** diagnostic and stays `status: failure`.
- The escalated budget is clamped by the ceiling; a blocked escalation skips the
  pass with a log line naming the blocker.
- Correct under `chunk_size` and `split`.
- `pytest -q` green, then **`budget-guardian` reports no findings.**

---

## Lane B — independent defects

**Verified against the code on 2026-09-03. Three of the five were already fixed
in the 0.7.0 squash (5dc02a3) and never should have been tickets.** They were
written from the "Hazards (still live)" list in the project notes, which was
authored *during* 0.7.0 development and not updated when the work landed. The
lesson is cheap and worth keeping: a ticket derived from a note is a hypothesis
until it is grepped.

| id | Status | Defect | Scope |
|---|---|---|---|
| ~~B1~~ | **Already done** | `_stop_server()` terminates, waits, force-kills, and leaves attached servers alone. Four behaviours pinned in `tests/test_lifecycle.py`. | — |
| B2 | **Live** | `RETRYABLE_ERRORS` is `(ConnectionError, httpx.TransportError)` and excludes `ollama.ResponseError` wholesale. Right for 400/404, wrong for **503/429** — a busy shared server is the retryable case by definition. Narrow the exclusion, don't widen the type. | `predictor.py`, `tests/test_failure_reporting.py` |
| ~~B3~~ | **Already done** | `test_required_params_accounts_for_every_task_config_field` and `test_required_params_has_nothing_that_comes_from_nowhere` are in `tests/test_lifecycle.py`, not `test_prediction_task.py` where the ticket looked. | — |
| ~~B4~~ | **Already done** | `_combine_results` calls `_combine_failures(rows, short_path.parent)`. | — |
| B5 | **Live** | `_TruncatingEmbeddings` clips examples to 2,000 chars before similarity selection with no log line. It changes *which* examples are chosen. | `predictor.py`, `tests/test_failure_reporting.py` |

> **Lane B is now one agent run.** B2 and B5 both touch `predictor.py`, so they
> go to the **same** `ticket-implementer`, in that order. There is nothing left
> to parallelise here.

---

## Lane C — naming and UX

**Parallel with Lane B; file-disjoint from it.**

### C1 — `Budget.source` must describe where the number came from

**Scope:** `budget.py`, `tests/test_budget.py` *(budget.py only — if `main.py`
needs to change, stop and hand it to Lane A)*

`resolve_budget` emits `"fitted to the data"` whenever `requested_ctx is None`,
which is true of a window that is three quarters user-set answer budget. Add a
source that distinguishes them, and make `describe()` say which term dominates.

**Acceptance:** a window whose `num_predict` exceeds its `prompt_tokens` does not
describe itself as fitted to the data; the existing describe tests still pass;
`budget-guardian` reports no findings.

### C2 — the Studio's three token fields

**Scope:** `gui.py`, `tests/test_gui_smoke.py`

"Context ceiling (tokens)" (`max_context_cap`), "Fixed context length (tokens)"
(`max_context_len`, hidden behind the *custom* radio option) and "Max output
tokens" (`num_predict`) sit in the same form with confusable names. The field a
user wants for the window takes two interactions to reach; the one that silently
changes the answer budget takes one. A real user set 20,000 in the wrong field
and it went unnoticed until the log was read closely.

Group them, say in each help string what the *other two* are not, and warn when
`num_predict` is set manually above a few thousand on a thinking model — that is
the shape of somebody trying to set the window.

**Acceptance:** a smoke test asserts the warning fires; help text names the
distinction; no behaviour change outside the Studio.

---

## Lane E — GPU verification

**Cannot be delegated. Needs a real model and a real card — Luc only.**

| id | What |
|---|---|
| E1 | Add an `effort` probe to `devtests/`: the same failing task at `think=true` / `"low"` / `false`, reporting fill rate and `eval_count`. This is the number that decides whether A1 alone is sufficient. |
| E2 | Re-run the RECIST task on this branch and compare the failure rate against the 7.8% baseline. |
| E3 | Probe whether Ollama passes `maxItems` / `maxLength` through to the grammar. llama.cpp supports them and "skips unsupported features silently", so this must be measured. It gates the deferred schema-bounds work. |

---

## Execution order

**Wave 1 — three agent runs, file-disjoint.** B2→B5 as one
`ticket-implementer` in sequence (shared file); C1 and C2 as one
`ticket-implementer` each. None edit `CHANGELOG.md` — they propose entries and
`docs-keeper` writes them in one pass when the wave lands. `budget-guardian`
reviews C1 before merge, since it is the only Wave 1 ticket in the sizing path.

Wave 1 was six runs before the Lane B verification pass; B1, B3 and B4 turned out
to be already shipped. Worth doing that check on any lane whose tickets came from
notes rather than from reading the code.

**Wave 2 — sequential, one session.** A1, A2, A3 in order, on the clean tree Wave
1 leaves behind. `budget-guardian` after A3. This is the release.

**Wave 3 — Luc.** E1, E2, then `docs-keeper` for the release notes.

The ordering is not arbitrary: Lane B touches `predictor.py`, and so does Lane
A's A2. Landing the small work first means Lane A starts from a clean tree
instead of merging into one.

---

## Deferred, with reasons

| Deferred | Why not now |
|---|---|
| **Two-pass extraction** (think freely with no grammar, then a short `think:false` formatting call) | The only way to genuinely decouple the two budgets given the upstream gap, and structurally the right answer. But it doubles calls per row and changes the pipeline shape. If E2 shows A1+A3 leave the failure rate materially above zero, this becomes 0.9.0's headline. |
| **Schema `maxItems` / `maxLength` bounds** | The most durable fix for output sizing — a declared bound becomes a hard decoder constraint and makes the estimate exact. Blocked on E3. |
| **Output calibration probe** (one real generation, size `num_predict` from `eval_count`) | Needs A2's `eval_count` capture to exist first, and needs the E1 numbers to choose a margin. Natural 0.9.0 work. |
| **Concurrency / `OLLAMA_NUM_PARALLEL`** | Designed already in the project notes. Interacts directly with A3 — a doubled retry window multiplied by four slots is eight times the KV cache — so it needs the retry pass to exist before it can be sized against it. Its own release. |
| **Partial JSON recovery** | Grammar-constrained truncation is a valid JSON prefix, so 12 of 15 lesions is recoverable. Widest correctness surface on the list, and A1+A3 may make it unnecessary. Last, or never. |
| **`_ASSUMED_LIST_ITEMS` / the 512 floor** | Real, but they size the *answer*, and the failures are dominated by *reasoning*. Fixing them now would move a number nobody has measured yet. A2 gives the measurement. |
| **GGUF tokenizer** | **Closed, not deferred.** Calibration on Dutch reported `1.08x` — 8% over on the language expected to drift worst. The estimate is sound. The only remaining argument is air-gapped deployment. |
