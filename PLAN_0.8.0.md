# 0.8.0 — bounding the chain of thought

**Branch:** `0.8.0`. **Reconciled against the code on 2026-09-03** — every ticket
below carries a file:line Evidence entry that was checked, not remembered.

## How tickets are written here

The first version of this plan had five Lane B tickets. Three were already fixed
in the 0.7.0 squash (`5dc02a3`) and only turned up when an agent was dispatched
to do one and correctly refused. They existed because the lane was written from
the "Hazards (still live)" note in project memory, which was authored *during*
0.7.0 development and never updated when the work landed.

So, the rule for this file:

> **A ticket carries Evidence: a file:line showing the current behaviour, checked
> against the working tree at the time of writing. No evidence, no ticket.**
> A note, a memory file or a changelog entry is a hypothesis about the code. The
> code is the only fact.

And the corollary, which is now in the implementer's contract: **before changing
anything, confirm the Evidence line still describes what you see.** If it does
not, stop and report. That refusal is a successful outcome.

---

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
and answer together, and thinking is spent first. Confirmed by Luc independently.

Not a defect in this package: [ollama/ollama#17561](https://github.com/ollama/ollama/issues/17561)
is an open proposal to bound thinking separately, and today there is no
`think_budget`. The only levers that exist are **shortening the trace**
(`think: "low"`) and **not thinking** (`think: false`).

Two supporting facts from the same run:

- `raw_output` is empty because it captures the *content* channel only. With
  `reasoning=True` the trace goes to `additional_kwargs["reasoning_content"]` and
  is discarded before the exception is raised.
- `context 33441 = 8441 prompt + 25000 answer + 0 headroom (fitted to the data)`
  described a window three quarters of which came from a user-set `--num_predict`.

---

## Lane A — the reasoning budget

**Sequential. One owner. Not delegated.** A1 changes how reasoning state resolves,
A2 changes what a failure carries, A3 consumes both. Three agents each holding a
third would reproduce, organisationally, the bug class this codebase exists to
have fixed. `budget-guardian` reviews the whole lane diff after A3.

### A1 — `--reasoning_effort {low,medium,high,max}`

**Evidence:** `prediction_task.py:119-128` resolves `_reasoning_param` to `True`,
`False` or `None` only; `grep -rn "reasoning_effort" llm_extractinator/` returns
nothing. `resolve_thinking` is at `run_config.py:279` and returns `bool`.

**Scope:** `main.py` (`parse_args` at :596, `TaskConfig`), `run_config.py`,
`prediction_task.py`, `gui.py`, `tests/test_run_config.py`, `tests/test_cli.py`,
`tests/test_prediction_task.py`

Ollama accepts `think: "low"|"medium"|"high"|"max"` on most reasoning models;
`ChatOllama`'s `reasoning=` takes those strings. gpt-oss accepts **only** levels
and silently ignores booleans, so the current three-state resolution is already
incomplete for that family.

Keep `resolve_thinking()` returning `bool` — the budget only needs to know
*whether* this run reasons — and add a separate resolver for the wire value:

```python
resolve_reasoning_param(detected, reasoning_model, no_reasoning, effort)
    -> True | False | None | "low" | "medium" | "high" | "max"
```

Precedence: `no_reasoning` beats everything; an explicit `effort` implies
reasoning is on. Follow `None = auto`; do **not** default to `"medium"`.

**Acceptance**
- `--reasoning_effort low` reaches `ChatOllama(reasoning="low")`, pinned by a test.
- `--no_reasoning --reasoning_effort high` → reasoning off, with a warning.
- Round-trips through `RunSettings.to_command()`.
- Present in `PredictionTask.REQUIRED_PARAMS`.
- `pytest -q` green.

**Scope decision: global flag only. Not per-model, not per-task — for now.**

*Per model, as a lookup table:* **no, and not later either.** There is no ground
truth for "qwen3 wants low, gpt-oss wants medium", it goes stale with every model
release, and it repeats the hardcoded-hardware mistake the model-sizing notes
already warn about. The one real per-model difference is a **capability** one —
gpt-oss accepts only levels and ignores booleans — and that is answered by asking
`model_capabilities()`, which A1 already does. Ask the server, never a table.

*Per task, as a `Reasoning_Effort` field in the task file:* **appealing, deferred,
with a stated trigger.** It is the motivating case — a classification task and a
per-lesion measurement task want different amounts of thinking, and the task file
is where task-shaped configuration belongs. Held back because:

1. The task file is a **public format**. Adding a field is easy; removing one
   after users have written task files is a breaking change. That needs E1, not
   plausibility.
2. It would create the **first setting that two places can supply**. Today the
   task file's vocabulary and the CLI's are disjoint. Overlap means a precedence
   rule, a resolver and tests for it — a small instance of the exact bug class
   `budget.py` exists to prevent, and worth paying for only once the knob is
   known to earn it.
3. **A3 likely subsumes it.** The retry pass already adapts effort per row from
   *observed failure* rather than *declared intent*: a task that does not need
   constraint never triggers it, one that does gets it with nothing configured.
   That is the same `None = auto` shape as the rest of the package, where auto
   means derived rather than guessed.

**Trigger to revisit:** E1 shows the right *default* level differs by task in a
way the A3 retry cannot reach. Then the task-file field is obviously right rather
than speculatively right, and it arrives with a precedence rule and a test.

### A2 — capture what the model actually did

**Evidence:** `predictor.py:204-212` is
`bound_llm.with_retry(...) | RunnableLambda(_strip_think_tags) | self.base_parser`.
The parser consumes the `AIMessage`, so `response_metadata` is lost.
`grep -rn "done_reason\|eval_count\|reasoning_content" llm_extractinator/` returns
only `prompt_eval_count` in `ollama_server.py:109-130`, which is the calibration
probe, not per-row generation.

**Scope:** `predictor.py`, `validator.py`, `tests/test_failure_reporting.py`

Classify **inside the chain** rather than correlating callbacks. `_strip_think_tags`
uses `model_copy`, so it preserves `response_metadata`. Insert a step after it and
before the parser that raises a typed exception carrying the metadata:

```
bound_llm | _strip_think_tags | _classify_completion | base_parser
```

- content empty after stripping → `EmptyCompletion`
- `done_reason == "length"` → `TruncatedOutput`
- otherwise pass through

The exception carries the metadata itself, so no run-id correlation is needed —
which is the trap here, since `chain.batch` runs rows concurrently and a parse
failure has no handle on the callback's `parent_run_id`.

New diagnostic columns on every row: `done_reason`, `eval_count`, and a clipped
tail of `reasoning_content`. **The reasoning tail is the point** — it is what
distinguishes a long-but-finite trace (A3 will recover it) from a repetition loop
(A3 will not). An ordered list of `eval_count` is enough for the distribution;
per-row correlation on *successful* rows is not worth restructuring the chain for.

**Acceptance**
- A faked `done_reason: "length"` yields `error_type: "TruncatedOutput"`.
- A faked empty content yields `EmptyCompletion`.
- Failure rows carry a non-empty reasoning tail when the fake supplies one.
- End of run logs p50 / p95 / max `eval_count` against `num_predict`.
- Successful rows still carry the full diagnostic key set as `None`.
- `pytest -q` green.

### A3 — the retry pass

**Evidence:** `grep -rn "_retry_phase\|attempts\|degraded" llm_extractinator/`
finds only unrelated prose. No second pass exists.

**Scope:** `prediction_task.py`, `main.py`, `utils.py`,
`tests/test_prediction_task.py`, `tests/test_pipeline_offline.py`

A **second pass over failed rows only**, not a per-row retry. Changing `num_ctx`
reallocates the KV cache, which means a model reload; per row that is ruinous,
once per pass it is cheap and it fits the existing phase structure.

```
main pass    num_predict = N        think = resolved
retry pass   num_predict = 2N       think = one level lower (or False)
```

Escalating both is deliberate. Budget alone loses to a repetition loop; reduced
thinking alone loses to a genuinely long trace. Together they cover both — and the
retry is a materially different request, which is what makes it legitimate under
the project's no-identical-retry rule.

- **Eligibility:** only `EmptyCompletion` and `TruncatedOutput`. Not transport
  failures, not genuine schema violations.
- **One pass. No ladder, no loop.**
- **Clamped:** the escalated window goes through `resolve_budget` under the same
  `max_context_cap` and `model_native_max`. If it cannot grow, skip and say so
  rather than reloading for nothing.
- **VRAM:** Ollama multiplies `num_ctx` by `OLLAMA_NUM_PARALLEL`. A doubled window
  on four slots is eight times the KV cache. Log the escalated window prominently.
- **No eligible failures → no reload.** The common case must cost nothing.
- **Marking:** recovered rows carry `attempts: 2` and `degraded: true`.
- Results merge by row index; correct under `chunk_size`, `n_runs` and `split`.

**Acceptance**
- Zero eligible failures → no reload, no second pass.
- Eligible failures → exactly those rows re-run and merged by index.
- Recovered rows carry `attempts` / `degraded`; first-pass rows do not.
- A row failing twice keeps its **second** diagnostic and stays `status: failure`.
- A blocked escalation skips the pass with a log line naming the blocker.
- Correct under `chunk_size` and `split`.
- `pytest -q` green, then **`budget-guardian` reports no findings.**

---

## Lane B — independent defects

**One agent run.** B2 and B5 both live in `predictor.py`, so they go to the same
`ticket-implementer`, in that order. There is nothing here to parallelise.

| id | Evidence | Defect |
|---|---|---|
| B2 | `predictor.py:44` — `RETRYABLE_ERRORS = (ConnectionError, httpx.TransportError)`; the comment at :36 excludes `ollama.ResponseError` for being a 400/404 | Right for 400/404, wrong for **503/429** — a busy shared server is the retryable case by definition. Narrow the exclusion by `status_code`; do not widen the type. |
| B5 | `predictor.py:77-88` — `_TruncatingEmbeddings` clips to `max_chars=2000` in both `embed_documents` and `embed_query`, no logging | The clip changes *which* examples the selector picks, silently. A debug line when it actually bites. |

**Already shipped in 0.7.0, not tickets** — recorded so they are not re-raised:
`_stop_server()` (`ollama_server.py:211`, four behaviours pinned in
`tests/test_lifecycle.py`); the `REQUIRED_PARAMS` coverage tests
(`tests/test_lifecycle.py:32,52` — note: *not* `test_prediction_task.py`);
split-mode failure merging (`_combine_results` calls `_combine_failures`).

---

## Lane C — naming and UX

Parallel with Lane B; file-disjoint from it and from each other.

### C1 — `Budget.source` must describe where the number came from

**Evidence:** `budget.py:150-152` — `source` is `"--max_context_len"` when
`requested_ctx` is set and `"fitted to the data"` otherwise, with nothing
consulting `output_tokens`. The docstring at :53 states only those two values.

**Scope:** `budget.py`, `tests/test_budget.py` *(budget.py only — if `main.py`
needs to change, stop and hand it to Lane A)*

A window that is three quarters user-set answer budget should not describe itself
as fitted to the data. Add a source that distinguishes them; make `describe()`
say which term dominates.

**Acceptance:** a window whose `num_predict` exceeds its `prompt_tokens` does not
report "fitted to the data"; existing describe tests still pass;
`budget-guardian` reports no findings.

### C2 — the Studio's three token fields

**Evidence:** `gui.py:944` "Cap context length", `:953` "Context ceiling
(tokens)" → `max_context_cap`; `:1107` "Context length strategy" radio, `:1121`
"Fixed context length (tokens)" → `max_context_len`; `:1054` "Max output tokens"
→ `num_predict`. All three in the same form, all measured in tokens.

**Scope:** `gui.py`, `tests/test_gui_smoke.py`

A real user meaning to set the window to 20,000 set the answer budget instead;
nothing in the interface or the log said so. The field for the window takes two
interactions to reach, the one that silently changes the answer budget takes one.

Group them; say in each help string what the *other two* are not; warn when
`num_predict` is set manually above a few thousand on a thinking model — that is
the shape of somebody trying to set the window.

**Acceptance:** a smoke test asserts the warning fires; help text names the
distinction; no behaviour change outside the Studio.

---

## Lane D — release mechanics

Do these last, in one pass, once Lanes A–C have landed.

| id | Evidence | What |
|---|---|---|
| D1 | `llm_extractinator/__init__.py` reads `__version__ = "0.7.0"` | Bump to `0.8.0`. It is the only place — `pyproject.toml` reads it dynamically. |
| D2 | `CHANGELOG.md:7` is `## [Unreleased]` | `docs-keeper` writes the collected entries there, then promotes the heading to `## [0.8.0] - <date>`. |
| D3 | — | Delete `PLAN_0.8.0.md`. It is an input to the work, not a record of it, and it rots the moment the work lands. |

**Merging to `main` publishes to PyPI** (`.github/workflows/publish.yml`).
Nothing else guards that, so D1 and D2 are not optional tidying.

---

## Lane E — GPU verification

**Cannot be delegated. Needs a real model and a real card — Luc only.**

| id | What |
|---|---|
| E0 | Settle the 163s suite time. `pytest -q --no-cov --durations=20`: time spread evenly is Windows/coverage, time concentrated in token-counting tests is a tiktoken network stall. Or measure directly: `python -c "import time; t=time.time(); import tiktoken; tiktoken.get_encoding('cl100k_base'); print(time.time()-t)"`. If the vocabulary cannot be fetched, every token count on that machine is a word count × 1.2 — which matters before E1. |
| E1 | Add an `effort` probe to `devtests/`: the same failing task at `think=true` / `"low"` / `false`, reporting fill rate and `eval_count`. **This is the number that decides whether A1 alone is sufficient.** |
| E2 | Re-run the RECIST task on this branch; compare against the 7.8% baseline. |
| E3 | Probe whether Ollama passes `maxItems` / `maxLength` through to the grammar. llama.cpp supports them and "skips unsupported features silently", so this must be measured. Gates the deferred schema-bounds work. |

---

## Execution order

**Wave 1 — three agent runs, file-disjoint.** B2→B5 as one `ticket-implementer`
in sequence (shared file); C1 and C2 as one each. None edit `CHANGELOG.md` — they
propose entries and `docs-keeper` writes them in one pass. `budget-guardian`
reviews C1, the only Wave 1 ticket in the sizing path.

**Wave 2 — sequential, one session.** A1, A2, A3 in order, on the clean tree Wave
1 leaves. `budget-guardian` after A3. This is the release.

**Wave 3 — Luc.** E0 first (it is free and it may invalidate E1's numbers), then
E1, E2. Then Lane D.

Wave 1 before Wave 2 because Lane B touches `predictor.py` and so does A2:
landing the small work first means Lane A starts from a clean tree.

---

## Deferred, with reasons

| Deferred | Why not now |
|---|---|
| **Two-pass extraction** (think freely with no grammar, then a short `think:false` formatting call) | The only way to genuinely decouple the two budgets given the upstream gap, and structurally the right answer. Doubles calls per row and changes the pipeline shape. If E2 shows A1+A3 leave the failure rate materially above zero, this becomes 0.9.0's headline. |
| **Schema `maxItems` / `maxLength` bounds** | The most durable fix for output sizing — a declared bound becomes a hard decoder constraint and makes the estimate exact. Blocked on E3. |
| **Output calibration probe** (one real generation, size `num_predict` from `eval_count`) | Needs A2's capture to exist, and E1's numbers to choose a margin. Natural 0.9.0 work. |
| **Concurrency / `OLLAMA_NUM_PARALLEL`** | Designed in the project notes. Interacts directly with A3 — a doubled retry window times four slots is eight times the KV cache — so it needs the retry pass to exist before it can be sized against it. Its own release. |
| **Partial JSON recovery** | Grammar-constrained truncation is a valid JSON prefix, so 12 of 15 lesions is recoverable. Widest correctness surface on the list, and A1+A3 may make it unnecessary. Last, or never. |
| **`_ASSUMED_LIST_ITEMS` / the 512 floor** | Real, but they size the *answer*, and the failures are dominated by *reasoning*. Fixing them now moves a number nobody has measured. A2 provides the measurement. |
| **GGUF tokenizer** | **Closed, not deferred** — unless E0 says the vocabulary cannot be fetched on the target machines, in which case it returns as the air-gapped fix. Calibration on Dutch reported 1.08x, so the estimate itself is sound. |
