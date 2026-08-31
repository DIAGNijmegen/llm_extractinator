# Development

## Setting up

```bash
git clone https://github.com/DIAGNijmegen/llm_extractinator
cd llm_extractinator
pip install -e ".[test]"
```

That is enough for the offline test suite. Running against a real model
additionally needs [Ollama](installation.md) and at least one model pulled.

---

## Two test suites, two different jobs

The distinction matters more than it looks, and knowing which one to reach for
saves a lot of time.

### `pytest` — the offline suite

```bash
pytest                     # the whole offline suite, a few seconds
pytest tests/test_budget.py -v
```

Every test here runs with the model **faked out**: no GPU, no Ollama, no model
downloads. That is what lets it run in seconds and in CI on every push.

It proves the *plumbing*. Prompts build, the schema binds, output parses, results
merge back onto the input rows, files land on disk, and the context window
arithmetic adds up. What it cannot prove is whether a real model produces good
extractions — the faked model returns whatever the test told it to.

A bare `pytest` deselects the `integration` marker, matching CI. To opt in:

```bash
pytest -m integration      # needs Ollama running and a model pulled
```

### `devtests/` — running against a real model

```bash
python devtests/run.py --model phi4
```

Ten probes, each covering a question the offline suite **structurally cannot
answer**, because a faked model always returns what it was told:

- Does grammar-constrained decoding really hold the model to an enum?
- Is the schema-derived `num_predict` actually enough for a wide schema?
- How far does the token estimate drift on Dutch?
- Do the field descriptions in the prompt change what comes back?

Pointing it at hardware:

```bash
# Manage Ollama automatically — start a server if needed, pull, stop afterwards
python devtests/run.py --model phi4

# Connect to a server that is already running; nothing started, pulled or stopped
python devtests/run.py --model phi4 --host http://localhost:11434

# Compare two models in one run
python devtests/run.py --model qwen3:4b --model phi4

# Three rows per dataset, for a quick shake-down
python devtests/run.py --model phi4 --limit 3

python devtests/run.py --list        # what each probe is for
```

`--host` is the only difference between local hardware and a shared or remote
box. A full sweep is around 100 generations — a couple of minutes on a small
model, longer on a large one.

It **reports numbers rather than asserting them**. There are no golden outputs
and no accuracy thresholds: a real model's output is not stable enough to pin,
and a threshold in a test only teaches you to ignore a red suite. The exit code
is non-zero only for hard breakage — an exception, or a probe where nothing
parsed at all.

Two runs can be compared directly, which is the point of the JSON report:

```bash
python devtests/compare.py devtests/reports/A.json devtests/reports/B.json
```

`devtests/README.md` has the full table of probes, an explanation of every
column, and the two numbers worth acting on.

### Which one do I want?

| I want to know… | use |
|---|---|
| Did I break the pipeline? | `pytest` |
| Does the context budget still add up? | `pytest` |
| Does a real model actually fill these fields? | `devtests/` |
| Is this schema too wide for the output budget? | `devtests/` |
| Did my prompt change help or hurt? | `devtests/` twice, then `compare.py` |
| Is the token estimate accurate for my language? | `devtests/`, read the `ratio` column |

---

## Conventions worth knowing

**Assert invariants, not numbers.** Token counts move whenever the prompt or the
estimator changes. Tests that pin exact figures break on every legitimate change
and get deleted; tests that pin a *relationship* — `prompt + num_predict <=
num_ctx` — survive and keep catching things. Where a test does need a figure, it
derives it from the live value rather than hard-coding it.

**`PredictionTask.REQUIRED_PARAMS` is an allowlist, and omission is silent.** A
setting added to `TaskConfig` but not listed there never reaches the task, with
no error. `tests/test_lifecycle.py` pins both directions, so adding a setting
forces a decision rather than defaulting to "not passed" by accident.

**Keep the docs in the same PR.** If you change task naming, required fields, or
CLI flags, update the docs alongside the code — a flag whose documented default
is wrong is worse than one that is undocumented.
