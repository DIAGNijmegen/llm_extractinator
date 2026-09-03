---
paths:
  - "tests/**"
  - "devtests/**"
  - "conftest.py"
---

# Tests

The suite is **hermetic**: 255 tests, roughly 7 seconds, no Ollama, no network,
no GPU. That is a property to defend, not an accident. A suite that grew slow
would stop being run before every change, and this codebase's bugs are the kind
only a fast suite catches.

Read `tests/conftest.py` and the file nearest your subject before writing a line.
Fakes for the model, the server manager and task files already exist — find them
rather than building your own.

## What belongs where

`tests/` gets everything a faked model can answer. `devtests/` gets **only**
questions a faked model structurally cannot, because it always returns what it
was told. That is a high bar and coverage is not a reason to clear it. Ten probes
live there today; adding an eleventh needs a justification of that shape.

`addopts` sets `--cov`, so **concurrent pytest runs collide on `.coverage`**. Use
`pytest -q --no-cov` when more than one agent or worktree runs the suite at once.

## What a good test looks like here

- It pins a behaviour someone relies on, and **the name is the specification**.
  `test_every_template_placeholder_is_supplied_by_the_caller` is the standard.
- It fails for exactly one reason.
- It uses real domain values where that is free. A 2,300-token report and a
  fifteen-item lesion list catch things `"foo"` and `[1, 2, 3]` do not.
- No golden outputs from a real model and no accuracy thresholds as assertions.
  A red suite people learn to ignore is worse than no suite. `devtests/` reports;
  a human judges. It exits non-zero only on an exception or a probe where nothing
  parsed.

## Known gaps worth closing

- Nothing checks that `PredictionTask.REQUIRED_PARAMS` covers every `TaskConfig`
  field. It is a silent-drop allowlist and the omission is invisible. Ticket B3.
- The context-budget baseline table in `tests/test_context_budget.py` was measured
  on a machine without the tiktoken vocabulary, so those are word-count figures
  and the file says so. Check which counter is live before comparing numbers
  across machines.
