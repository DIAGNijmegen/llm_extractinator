---
description: The definition of done for a ticket on this branch. Run before reporting any ticket complete, and before opening a PR. Checks the suite, the invariant, the allowlist, the changelog and the scope of the diff.
allowed-tools: Bash(pytest:*) Bash(git diff:*) Bash(git status:*) Read Grep Glob
---

Run these in order and report each result. Stop at the first hard failure and
say which one failed — do not continue and do not summarise a partial pass as
success.

## 1. The suite

```bash
pytest -q
```

Hard gate. 255 tests, roughly 7 seconds. Any failure or error stops the check.
A test that is slow is itself a finding: something reached the network, and the
most likely culprit is a tiktoken vocabulary fetch (see the token-counting
notes — `lru_cache` does not cache exceptions, so a failed fetch is re-attempted
per row).

## 2. Scope of the diff

```bash
git status --porcelain
git diff --stat
```

Every changed file must be named in the ticket's scope. An unexpected file is a
hard failure — parallel agents are working in this repo, and an out-of-scope
edit is somebody else's merge conflict. Report the file and stop.

## 3. The allowlist

If the diff adds a field to `TaskConfig`, confirm the same name appears in
`PredictionTask.REQUIRED_PARAMS`. It is a silent-drop allowlist: a missing entry
produces no error, just a setting that never reaches the model.

## 4. The invariant

If the diff touches `budget.py`, `prediction_task.py`, `data_loader.py`,
`output_parsers.py` or the sizing path in `main.py`, this check is not
sufficient on its own — dispatch the `budget-guardian` agent and include its
report. Do not self-certify a change to the sizing path.

## 5. The Evidence line

Confirm the report says whether the ticket's Evidence line matched the working
tree. A ticket completed without that check is not verified, however green the
suite is — three tickets in the first version of this plan described defects that
had already been shipped.

## 6. Tests and changelog

- Every behavioural change in the diff has a corresponding test in `tests/`.
  Name it.
- The entry for `CHANGELOG.md`'s `## [Unreleased]` heading was *proposed in the
  report*, not written into the file — implementers do not edit it, `docs-keeper`
  writes them in one pass.
- If a flag, default or documented log message changed, `docs/cli.md` and
  `docs/settings-reference.md` are consistent with it — or the `docs-keeper`
  agent has been dispatched.

## 7. Report

One line per check: name, pass or fail, and the evidence (the pytest line, the
file list, the test name). If everything passes, say so and name the ticket.
