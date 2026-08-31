# Dev tests — running against a real model

> Part of the project's testing setup — see
> [Development & testing](../docs/development.md) for how this sits alongside
> the offline `pytest` suite, and which of the two answers which question.

The suite in `tests/` fakes the model out entirely. That is deliberate: it makes
the plumbing testable in seconds with no GPU. But it means a whole class of
question is structurally unanswerable there, because a faked model always
returns exactly what it was told to.

This harness exists for those questions, and only for those:

- Does grammar-constrained decoding actually hold the model to an enum?
- Is the schema-derived `num_predict` really enough for a wide schema?
- How far does the token estimate drift on Dutch?
- Do the field descriptions in the prompt change what comes back?

Anything a faked model could verify belongs in `tests/`, not here.

## Running it

```bash
# Let the tool manage Ollama itself — starts a server if needed, pulls, stops
python devtests/run.py --model phi4

# Point at a server that is already running (nothing started, pulled or stopped)
python devtests/run.py --model phi4 --host http://localhost:11434

# Compare two models in one go
python devtests/run.py --model qwen3:4b --model phi4

# A quick shake-down: three rows per dataset
python devtests/run.py --model phi4 --limit 3

# One or two probes only
python devtests/run.py --model phi4 --only dutch,wide

python devtests/run.py --list
```

`--host` is the only difference between local hardware and a remote box. Without
it you get the normal managed lifecycle; with it the run only ever connects, so
it is safe against a machine somebody else is using.

A full sweep is around 100 generations. On a small model that is a couple of
minutes; on a 35B it is longer, and `--limit` is there for when you just want to
know whether anything is on fire.

## What each probe is for

| probe | data | what only a real model can tell you |
|---|---|---|
| `basic` | 10 short EN reports | control — if this fails, ignore everything else |
| `enums` | 10 short EN reports | does the grammar hold the model to the allowed values |
| `wide` | 4 long reports | 21 fields: is the derived output budget actually enough |
| `nested` | 10 short EN reports | list-of-objects under the grammar |
| `optional` | 10 short EN reports | does it *omit* absent fields, now that `required` is not forced |
| `reasoning` | 10 short EN reports | `format` + `think` together; needs a judgement, not a copy |
| `dutch` | 10 short NL reports | token-estimate drift on non-English text |
| `long` | 4 long reports | context sizing on ~2,000-token documents |
| `described` | 10 short EN reports | A/B: six fields **with** descriptions |
| `bare` | 10 short EN reports | A/B: the same six fields **without** descriptions |

`wide` and `long` share a dataset on purpose — one has a wide schema and the
other a narrow one, so the pair isolates schema width from document length.
`described` and `bare` are identical but for the descriptions, for the same
reason.

## Reading the table

```
probe       rows   ok fail  num_ctx  n_pred measured  ratio   fill   secs
basic         10   10    0      819     512      478  1.089  0.933    41.2
```

| column | meaning |
|---|---|
| `ok` / `fail` | rows that parsed into the schema, and rows that did not |
| `num_ctx` | the context window the resolver chose |
| `n_pred` | the generation budget — derived from the schema unless overridden |
| `measured` | what the model itself counted for the longest prompt |
| `ratio` | reserved ÷ measured. **1.0 means the estimate is exact** |
| `fill` | of the values a successful row could carry, how many are non-null |
| `secs` | wall time for that probe |

`fill` is a crude proxy for quality, but it is the right shape for a comparison:
two runs over the same data with the same schema differ only in how much the
model actually found.

## The two numbers worth caring about

**`ratio` on `dutch` versus `basic`.** The token estimate uses OpenAI's
`cl100k_base` against Qwen and Phi models and cannot see the chat template. Near
1.0 on both means it is sound and there is nothing to fix. A large gap on Dutch
is the argument for reading the model's real vocabulary out of
`/api/show?verbose=true` — work that has been deliberately deferred pending
exactly this measurement.

**`fill` on `described` versus `bare`.** The schema's field descriptions did not
reach the model at all before 0.7.0 — they went to Ollama inside the `format`
parameter, which is compiled to a grammar that keeps the structure and discards
the prose. This pair is the only way to find out what putting them in the prompt
was worth. Read a sample of both outputs before concluding: fill rate rewards a
model that answers confidently, which is not the same as answering correctly.

## What this deliberately does not do

**No golden outputs.** A real model's output is not stable enough, and pinning it
would turn "the model changed" into "the tests are red".

**No accuracy thresholds as assertions.** Same reason. The harness reports; you
judge. It exits non-zero only for hard breakage — an exception, or a probe where
nothing at all parsed.

## Comparing runs

Every run writes `devtests/reports/<timestamp>.json`.

```bash
python devtests/compare.py devtests/reports/A.json devtests/reports/B.json
```

Two models, or the same model before and after a change. Movement, not verdicts.

## The data

Entirely synthetic, generated by `_make_data.py` — invented findings in the shape
of radiology reports, in English and Dutch, plus four long staging reports whose
length comes from an enumerated lesion table, which is what makes real ones long.
No real report, patient or identifier appears anywhere.

Regenerate with `python devtests/_make_data.py` and `python devtests/_make_tasks.py`.
The generated files are committed: re-rolling the inputs between two runs would
make the comparison meaningless.

## One caveat

The harness reads `num_ctx`, `n_pred` and the calibration out of the run's log
records, using the `%`-style arguments rather than parsing prose. That is a
deliberate coupling to log messages — fine for a dev tool, but if a column reads
`n/a` when you did not expect it, a message has probably been reworded.

There was a second cause, now fixed, that is worth knowing about because it
bites anything that captures this package's logs: `extractinate` configures
logging with `logging.basicConfig`, which is a **no-op once the root logger
already has a handler**. Attaching a capture handler before the call therefore
left the root logger at `WARNING`, and every `INFO` record — which is all of the
interesting ones — was discarded before any handler saw it. The run itself
worked perfectly and every column read `n/a`. `run_probe` now sets the root
level explicitly, and attaches its own per-probe file handler, since
`basicConfig` being a no-op also means the run's own log file is never created.
