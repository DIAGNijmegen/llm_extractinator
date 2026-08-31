# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.7.0] - 2026-08-31

### Upgrading from 0.6.x

Every interface is unchanged: the same CLI flags, the same `TaskConfig` fields,
the same `extractinate(**kwargs)`, and task files parse as they did. Nothing will
fail to start. What follows are the behaviour changes that will be felt anyway.

- **Failed rows contain `None`, not type defaults.** The one documented breaking
  change; see above. Row shape is unchanged, so a `DataFrame` built from the
  output keeps its columns — but code that reads a field without first checking
  `status` may now meet `None` where it used to get `0`, `""` or a plausible
  `Literal`.

- **Three new columns on every row.** `error_type`, `error_message` and
  `raw_output` are present on successes too, as `None`, so that a predictions
  file has one uniform set of columns. Anything validating the output with
  `extra="forbid"`, or asserting a column count, needs updating.

- **`extractinate` now raises `ContextBudgetError`.** It previously caught bare
  `Exception` and returned `None` — it never raised. A configuration whose
  generation budget cannot fit its context window now stops the run instead of
  being reported as a completed one. Scripted callers that relied on it never
  raising should wrap it.

- **VRAM demand can rise, by design.** In 0.6.x a thinking model sized its window
  for `num_predict=512` and then raised the budget to ~5,500 behind the window's
  back; the window never grew and Ollama truncated the prompt in silence. The
  allowance is now inside the sizing, so the same run legitimately asks for
  roughly 5,000 more tokens of window. Add the schema-derived `num_predict` — a
  21-field schema gets ~1,200 rather than 512 — and a configuration that fit on a
  card before may not now. `--max_context_cap` bounds it and reports what the cap
  costs.

- **Configurations that used to run may now refuse to start.** `--max_context_len
  N` smaller than `num_predict` reached Ollama unchallenged before and was
  silently truncated; it is now rejected with a message naming the setting at
  fault. Translation with a modest fixed window was the worst case.

- **Two module-level functions were removed:** `validator.handle_failure` and
  `validator.validate_results`. `handle_prediction_failure` also lost its unused
  `input_data` argument. These were internal in spirit but importable in
  practice.

- **Extraction output is not comparable with 0.6.x.** The prompt changed
  materially: field descriptions are now included, the "think step by step"
  instruction is gone, and the missing-value rule was rewritten. Results on
  identical inputs will differ. If you have published or recorded numbers from
  0.6.x, re-run rather than comparing across the boundary.


- Rebuild the Studio's Run tab around what a run actually varies. Task, model and scope now sit in one row at the top with the Run button; everything else moved into a single collapsed **Settings** panel grouped by the question it answers — Hardware, Generation, Prompt, Execution — replacing *Repetition & logging* / *Prompting & context* / *Sampling*, which had put the context ceiling and the output budget in different sections despite their being two halves of one window. The button used to be the last element on the page, below three sections and a ten-control expander
- Show a run instead of a log. Launching now replaces the form with a run view: progress by rows, elapsed time, the resolver's budget line promoted out of the transcript, and the log behind a toggle. Leaving twenty controls on screen under a scrolling log meant that during the one period when nothing can be changed, the page showed everything that could be
- **Fixed:** there was no way to stop a run. The tab iterated `process.stdout` on Streamlit's script thread, which parked the whole app — no other tab responded, and a click could not be processed at all, so a Stop button was not merely missing but impossible. The subprocess is now drained by a daemon thread and the page polls a fragment, so Stop works and the rest of the Studio stays live. Stop terminates rather than kills, giving `extractinate` the chance to shut down the Ollama server it started
- **Fixed:** the Studio's *Max tokens to generate* field showed a number the run did not use. `RunSettings.num_predict` was `int = 512`, with 512 doubling as the "do not emit the flag" sentinel — so the field read 512, the command omitted it, and the backend derived something else from the schema, often 1,204. It is now `Optional[int] = None` behind a *Set max output tokens manually* checkbox, the same shape temperature already had, and the review strip reads `Auto (from schema)`
- **Fixed:** selecting a thinking model made the Studio advise, in its default state, raising max output tokens to 5,512 — a value that would have pinned the budget and switched off the schema sizing it was meant to improve. `suggested_num_predict` gained the reasoning allowance in 0.7.0, and the caption had been comparing against it. The backend adds that allowance itself; the caption is gone
- Disable **Run** when the configuration cannot work, rather than rendering an error beside a live button. The backend now refuses these outright, so the click bought a traceback
- Apply a hardware preset through a deferred write at the top of the next rerun. Streamlit rejects a session-state write to a widget that has already rendered, so the preset button previously had to sit above every field it filled — which is why the model picker and the context ceiling could not be placed by meaning. The lasting "✅ Applied Medium" caption is also gone: it never cleared, so it kept asserting a state that stopped being true at the first hand edit
- Thin out the Studio's emoji. Eleven of them across section headers, buttons, captions and status was most of the visual noise; they are now navigation landmarks on the tab bar and warnings, nothing else

- **Fixed:** rebuilding the Docker image did not update Ollama, so a newly released model stayed unsupported in an image you had just rebuilt. `RUN curl … install.sh | sh` is cached by its command text, which never changes — and the layer sits above `COPY . /app`, so not even a source change invalidated it. Ollama is now installed through an `OLLAMA_VERSION` build argument, and a new `build.sh` resolves the newest release and passes it in, so the layer rebuilds exactly when the version moves. Setting `OLLAMA_VERSION` pins instead, to roll back or reproduce an older image. Documented in [Docker](https://diagnijmegen.github.io/llm_extractinator/docker/), which previously did not mention building the image at all

- Document how to run the tests, which nothing did before — not the README, not `docs/`, not the mkdocs nav. A new [Development & testing](https://diagnijmegen.github.io/llm_extractinator/development/) page covers both suites and, more usefully, which one answers which question: `pytest` proves the plumbing with the model faked out and cannot say anything about extraction quality, while `devtests/` covers only what a faked model structurally cannot. Linked from the README's contributing section, the docs index, the nav, and back from `devtests/README.md`

- Add `devtests/`, a harness for running against a real model. Ten probes, each covering a question the offline suite structurally cannot answer because a faked model always returns what it was told: whether the grammar really holds the model to an enum, whether the schema-derived output budget is actually enough, how far the token estimate drifts on Dutch, and whether the field descriptions now in the prompt change what comes back. `python devtests/run.py --model phi4` manages Ollama itself; `--host` points at a server that is already running and touches nothing. It reports numbers rather than asserting them — no golden outputs, no accuracy thresholds — and exits non-zero only for hard breakage. Data is synthetic throughout

- **Fixed:** `--max_context_len split` returned rows ordered by document length rather than by input position. The two halves are processed separately and concatenated, so every short row came back before every long one — silently breaking a positional join onto the source data, which is the same failure chunked runs had before their merge was ordered by row offset. Rows are now restored to input order
- **Fixed:** a `split` run left `failures-short.json` and `failures-long.json` behind while merging and deleting the two prediction files, so the row numbers in them referred to files that no longer existed. One `failures.json` is now written for the merged run
- Shut down the Ollama server this tool started. `stop()` ran `ollama stop <model>`, which unloads the model from VRAM but leaves the server process running — and since we launched it, nothing else ever would. A server that was already up, or one named with `--ollama_host`, still belongs to somebody else and is left alone
- Pin `PredictionTask.REQUIRED_PARAMS` against `TaskConfig`'s fields. It is an allowlist and omission is silent: an unlisted field never arrives, which is how `max_context_cap` failed to reach the task for a whole release. Adding a setting now forces a decision rather than defaulting to "not passed" by accident

- Correct the context window from the measurement instead of refusing the run. When the prompt turns out larger than the window reserved for it, the measured count is *exact*, so the window is re-resolved from it under the same ceilings and the model reloaded — turning "your estimate was wrong, go and change a flag" into a window that is simply right. An explicit `--max_context_len` is still never overridden, and a hardware cap or the model's own limit can still leave no room to grow; those cases stop the run naming what blocked it
- Size `--num_predict` from the output schema when it isn't given. It was a number every user had to invent, and one value had to serve a schema of three enum fields and a schema of thirty free-text ones — a thirty-field schema needs roughly 1,360 tokens where the old fixed default gave 512, and a truncated answer loses the whole row rather than degrading. The derived value is floored at the previous default so auto-sizing can only raise the budget, never lower it; the reasoning allowance is added on top. Passing `--num_predict` still overrides. The Studio needs no change: it only emits the flag when you move it off its default, so leaving it alone now means "size it for me"

- Put the schema's field descriptions into the prompt. They were being collected and then thrown away: the schema goes to Ollama as the `format` parameter, which is compiled into a grammar that keeps the structure — types, allowed values, which fields are required — and discards every `description`. A field documented as "the primary diagnosis, verbatim from the report" arrived at the model as the bare key `diagnosis`. The system prompt now lists each field with its type, its allowed values where it has a fixed set, whether it is optional, and the description the task author wrote
- Stop asking the model for things the grammar forbids. The system prompt ended with "Think step by step", but every request binds `format=<schema>`, so the first token must be `{` and there is nowhere for reasoning to go — a thinking model reasons in its own channel and does not need to be asked. It also said to return `null` or `"N/A"` for anything missing, which for a required integer the grammar permits neither of, so the model had to emit a fabricated number instead. That rule is now stated in terms of whether the field is actually optional
- **Fixed:** a schema whose fields were all marked `optional` was forced to be entirely mandatory. `predict` set `required` to every property whenever pydantic omitted the key — and pydantic omits it precisely when every field has a default, i.e. exactly the all-optional case. The model was obliged to invent values for the fields it had been given permission to leave out
- **Fixed:** a task file with no `Description` rendered the literal string `None` into the middle of the system prompt. It now falls back to a generic instruction and warns, since describing the task is the single biggest influence on extraction quality
- Add an optional `Extra_Instructions` field to the task file, appended after the built-in rules. `Description` was the author's only lever and it lands mid-template among instructions they can neither see nor override; this is an explicit place for guidance that applies to the whole task — an output language, a unit convention, a house style

- Check the token estimate against the model's own count before spending a run on it. Ollama still has no tokenize endpoint (ollama/ollama#3582 closed without landing one, PR #12030 open), but every chat response reports `prompt_eval_count` — the exact number of tokens the server evaluated, through the model's real tokenizer *and* its real chat template, neither of which a local estimate can see. Once the prompt is assembled, the longest one is sent with `num_predict: 1` and the count compared against the window reserved for it. The run reports the ratio, so the estimator's accuracy is finally a measured number rather than a guessed 15% margin
- Stop a run whose prompt is larger than the window reserved for it, instead of letting Ollama silently drop the beginning of every long document. This is the case the safety margin exists to prevent and could not previously detect — tokenizer drift is worst on non-English text, which is exactly where the estimate is least trustworthy. The message says by how much and which setting to change
- Treat `prompt_eval_count` as suspect rather than authoritative: a count that cannot be a plausible tokenization of the text is discarded with a warning. It has a history of being wrong in the one direction that matters (ollama-python#271 reports it stuck at 1026 tokens across prompts of 57,000 characters), and an under-reported count would make the estimate look safer than it is. A calibration that cannot be performed — no server, an implausible answer — leaves the estimate unverified and does not stop the run

- Resolve the context window and the generation budget in one place, from inputs that are final. `budget.resolve_budget` is now the only thing that produces `num_ctx` and `num_predict`, and it enforces `prompt + num_predict <= num_ctx <= min(--max_context_cap, the model's native context)`. The rule previously lived in four places — `TaskRunner._context_len` sized the window, `PredictionTask.__init__` raised `num_predict` afterwards, `TaskConfig.__post_init__` validated the cap against it and `DataLoader.adapt_num_predict` raised it for translation — each knowing part of the relationship and none knowing all of it
- Reorder the run so the model is inspected before anything is sized: start the server, pull, read `/api/show`, *then* size the window. This retires the reasoning-allowance bug at its root — the allowance is now part of what gets sized rather than something added behind the window's back — and it means the server is started, pulled and stopped once per run instead of once per phase, so `--max_context_len split` no longer unloads the model from VRAM between the short and long halves only to reload it
- Read the model's native context length from `/api/show` and treat it as a ceiling. A window larger than the model supports is not a larger window; nothing read this before, because the window was sized before the model was available to ask
- **Fixed:** `--max_context_len N` was never validated against `--num_predict`, so a fixed window smaller than the generation budget reached the model unchallenged and Ollama truncated the prompt silently. Translation was the worst case — it reserves the longest input plus a 5000-token buffer, so a run with a modest fixed window was asking for a ~5000-token answer inside a 512-token window. Both are now refused, with a message naming which setting made the window that size
- A capped window is clamped rather than rejected, but says what the cap costs: "capped at 4096 by --max_context_cap, 79 short of the 779 this data wants. 12 of 5000 documents exceed the 3500 tokens that leaves for input and will be truncated." The previous warning said only that "the longest inputs will be truncated", which does not distinguish an acceptable trade from a ruined run
- Log the budget as one line — `context 4096 = 1119 prompt + 512 answer + 2465 headroom (fitted to the data)` — so that the prompt and the answer are visibly competing for the same space

- **Breaking:** a failed row no longer carries invented values. Every schema field is `None` instead of a type default, and `Literal` fields are no longer filled with `random.choice` over the permitted values — which produced a plausible, non-reproducible answer that nothing downstream could tell apart from a real extraction. `0` was the dangerous case: a failed heart-rate row was a real zero in a mean, where `None` becomes `NaN` and drops out. Rows keep every schema key, so the output *shape* is unchanged; only the values are
- Record why a row failed. Failed rows gain `error_type`, `error_message` and `raw_output` — the model's own text, which `handle_prediction_failure` previously accepted as an argument and discarded. Successful rows carry the same keys as `None`, so a predictions file has one uniform set of columns. A run with failures also writes `failures.json` next to its predictions, and the run log ends with a breakdown by exception type rather than a bare count: "the server went away" and "the model wrote prose" need opposite responses
- Retry only what a retry can fix. `.with_retry()` wrapped the whole chain and retried on `Exception`, so every parse failure was re-sent three times — byte-identically, at temperature 0, for three identical failures at three times the cost. The retry now wraps the model call alone, and fires only on `ConnectionError` (ollama's translation of `httpx.ConnectError`) and `httpx.TransportError`. An `ollama.ResponseError` — a 400 or a 404 — is not retried, because it will be the same next time
- Count progress in rows rather than in calls to the model. `on_llm_end` fires once per call, so a retried row ticked the bar more than once and five rows could report `15it` against a total of 5 — reading as "everything passed" immediately before the rows turned out to be empty. Rows are now identified by their parent run, and `on_llm_error` is handled, so a row whose call failed advances the bar instead of leaving it silently short
- A failed translation keeps its original text. The failure default was written straight into the document, so an untranslatable row was silently blanked and then extracted from nothing. The run now continues on the untranslated original, and says how many rows that happened to
- Remove the unused output-fixing machinery: `prompt_templates/output_fixing/`, and `validator.validate_results`, which was referenced only by its own tests. With `format=<schema>` bound on every call decoding is grammar-constrained, so structurally invalid JSON is close to impossible; the realistic failure is truncation, and a repair prompt — the original instructions plus the broken completion — is strictly longer than what already overflowed

- Stop re-attempting a failed tokenizer load on every row. `_get_encoding` is `lru_cache`d, but `lru_cache` does not cache *exceptions*, so when `tiktoken.get_encoding` failed the lookup ran again for every single row — and its failure path is an HTTP fetch of the vocabulary from `openaipublic.blob.core.windows.net`, meaning an offline or firewalled machine paid a connection timeout per row while silently falling back to a word count. It now returns `None` on failure, which the cache does keep, so the attempt happens once per process. Measured on the test suite: 158s to 11s

- Refuse a run whose generation budget cannot fit in its context window. `num_ctx` is the whole window — the prompt and the generated answer share it — so `num_predict >= num_ctx` leaves nothing for the input, and Ollama does not object: it truncates the prompt silently and returns worse output with no indication why. The check runs where the two numbers actually reach the model, so a value that bypassed the sizing logic is still caught, and a failing configuration now raises rather than being logged and reported as a completed run. Today this catches the auto-detected thinking path, where the reasoning allowance is added to `num_predict` *after* the window has been sized; the error names `--reasoning_model`, which sizes the window with the allowance included
- Add an end-to-end context-budget test matrix — thinking or not × `max`/fixed/`split` × full run or test run × zero- or few-shot — asserting `prompt + num_predict <= num_ctx` against the prompt the model is actually given rather than against the estimator's opinion of it. It uses a new `record_model_kwargs` fixture that observes `ChatOllama` construction instead of `PredictionTask` construction: the earlier boundary cannot see mutations made inside `PredictionTask.__init__`, which is why the ordering bug survived a suite that already covered context sizing. The twelve thinking cells are marked `xfail(strict=True)`, so they turn the suite red the moment they start passing


- Choose `--temperature` automatically when it isn't set: `0.0` for an ordinary model, `0.6` for a thinking one. Qwen's model cards are explicit that thinking mode must not run greedy — *"DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions"* — and a repetition loop here spends the whole `--num_predict` budget and returns no JSON at all. An explicit `--temperature 0` still forces greedy, with a warning
- Rework the context-window estimate. It now measures what the prompt actually contains — the system prompt with the task description substituted in, the longest document, and the largest few-shot exchanges that could be selected (answers included) — plus a percentage safety margin for tokenizer drift. Previously it multiplied the longest *document* by `num_examples + 1`, so five examples reserved six times the longest document and spent the VRAM to match, while the prompt scaffolding went uncounted behind a flat 1000-token buffer
- Count few-shot example outputs, not just inputs: both halves of an exchange go into the prompt
- Cache the tokenizer instead of rebuilding it per row, and warn once (rather than silently) when the tokenizer cannot be loaded and the estimate falls back to a word count

- Replace the Studio's *Reasoning model* toggle with an **Auto / On / Off** control defaulting to Auto. The toggle seeded itself from capability detection, which froze whatever was knowable at seeding time: applying a hardware preset before entering an Ollama server URL meant detection ran with no server, answered "not a thinking model", and never re-ran — so the Studio then sent `--no_reasoning` for a reasoning model and reported it as the user's choice. Auto asks the question at run time instead, where the model has actually been pulled
- Correct the hardware preset context ceilings, which were round numbers rather than computed ones. High is 262,144 (not 32,768) and needs 40GB — `qwen3.5:35b` is a 35B-A3B MoE whose hybrid attention keeps a KV cache on only 10 of 40 layers, so context is cheap but its 22.4GiB of weights will not fit a 24GB card; Low is 16,384 (not 4,096). Medium stays at 8,192. Presets no longer set `num_predict`: output length follows the schema and whether the model thinks, not the size of the card
- Resolve a preset's model against the Ollama server once one is known, so entering a server URL after applying a preset selects the model in the dropdown instead of leaving it pinned to *Other (enter below)*
- Reject `--max_context_cap` at or below `--num_predict`: `num_ctx` is the whole window, so such a ceiling leaves no room for the prompt
- Align the Studio's paired controls that mixed a labelled widget with an unlabelled one (preset + Apply, test-run rows + random sample, context cap, schema file + build)

- Restructure the Studio Run tab into Model / Scope / Advanced settings / Review. Test run and chunking now sit together under Scope rather than on opposite sides of the Advanced boundary, section headings are consistent, and a review strip above **Run** shows the settings that will actually be used — which is also what makes a preset's effect on the collapsed Advanced fields visible
- Add a "Recommended settings" picker to the Studio Run tab (High/Medium/Low VRAM) that fills in the model, max tokens to generate and context ceiling on **Apply**, with a caption stating what each tier sets. It is an action rather than a mode, so it cannot go on claiming to describe a config that has since been edited by hand
- Move the Run tab's command construction out of `gui.py` into `run_config.RunSettings`, so the one piece of the Studio with real logic in it is unit-testable; add AppTest smoke tests covering the tab
- Add `--max_context_cap` to bound the context window under `--max_context_len max`/`split`: the window is still fitted to the data, it just never grows past the cap. The context window, not the model weights, is usually what pushes a run out of VRAM
- Size a test run's context window from the full dataset rather than the sampled subset, so `--test_run_size` previews the run it is meant to preview instead of a smaller one that may not represent it
- Add `--test_run_size`/`--test_run_random` to run on a subset of the test set instead of the full dataset, surfaced in Studio as a "🧪 Test run" toggle
- Add "Process in chunks" option to the Studio Run tab, exposing the existing `--chunk_size` CLI flag
- Add `--no_reasoning` to force reasoning mode off, sending Ollama an explicit `think: false` rather than merely declining to ask for thinking (which leaves an always-on model reasoning). Auto-detection previously could not be overridden — `reasoning_model=False` was ignored on any model advertising a `thinking` capability, so the Studio's *Reasoning model* toggle silently did nothing when switched off
- Fix `--chunk_size` runs merging their chunks in filesystem order, which scrambled row order in the combined output relative to the input (chunk files are named by row offset, so even a lexical sort put row 100 before row 50)
- Fix a repeated test run at a different row count returning the earlier run's output: the row count is now part of the output folder name (`<task>-test<N>-run<idx>`), so sizes no longer collide under the skip-if-exists guard
- Move the run-folder naming and chunk-file collection into `utils.py` so the writer (`PredictionTask`) and the Studio's Results tab share one definition instead of two that can drift
- Correct the Studio's *Overwrite existing files* help text: `--overwrite` re-runs and replaces existing output, it never deletes the run folder
- Fix Streamlit logging "widget was created with a default value but also had its value set via the Session State API" for the task picker, model name, reasoning toggle, max-tokens and text-column widgets; each now seeds through session state instead of passing a default alongside a key
- Make a bare `pytest` skip the Ollama-dependent integration test by default, matching what CI already does; run `pytest -m integration` to opt in

## [0.6.0] - 2026-07-22

- Redesign Studio with a shared brand theme (`theme.py`): branded header, dark sidebar with workflow guide, card-styled metrics/expanders, packaged logo assets; schema builder matches the same look
- Reorganize and expand documentation: grouped mkdocs nav (Getting started / Building tasks / Running / Reference / Deployment), new Quickstart, Understanding Output, and Troubleshooting pages
- Add `--ollama_host` to connect to an already-running Ollama server (local on a different port, or remote) instead of always spinning up and managing one
- Rewrite test suite as hermetic offline pipeline tests, with the real-model check moved behind an opt-in `integration` marker
- Fix `seed=0` being silently ignored in `extractinate` (`if config.seed:` treated 0 as falsy)
- Route error output through `logging` instead of `print`/`traceback.print_exc()` so it reaches the log file
- Remove a duplicate import block in `data_loader.py`
- Migrate packaging metadata from `setup.cfg` to `pyproject.toml` (PEP 621); single-source the version via `__version__`; bump `python_requires` to `>=3.10`
- Add `CITATION.cff` for the JAMIA Open paper

## [0.5.14] - 2026-05-27

- Fix structured output for always-on reasoning models (e.g. qwen3.5)
- Auto-detect thinking models and improve reasoning/overwrite UI
- Fix Ollama server startup noise and route verbose logs to file
- Add gitkeep files so runtime folders exist on a fresh clone

## [0.5.13] - 2026-04-03

- Fix Studio GUI saving files to the wrong directory and text column instability

## [0.5.12] - 2026-04-03

- Fix embedding context overflow and duplicate PyPI publish

## [0.5.11] - 2026-04-02

- Fix `split_data` with few-shot examples
- Overhaul the Studio GUI

## [0.5.10] - 2026-02-24

- Add a CI workflow and fix the Dockerfile for the updated Ollama install
- Update GUI, utils, and data loader; add tests
- Refresh README with project logo and details

## [0.5.9] - 2025-12-09

- Multiple improvements and bug fixes
- Expand docs, including CPU-only setup instructions

## [0.5.8] - 2025-11-14

- Add Docker support and improve the Dockerfile
- Fix the `reasoning_model` flag and bugs from the LangChain 1.0 upgrade
- Make prompts more robust to brackets; modernize prompt utils
- Show the Ollama model-pull progress in its own UI field
- Documentation and README improvements

## [0.5.5] - 2025-09-19

- Add a standalone launcher GUI; general GUI improvements
- Migrate to LangChain's `with_structured_output` and clean up code
- Fix an output-fixing parser issue and an input field bug
- Default `max_context_len` to `"max"`; ASCII-friendly progress bars
- Rename/clean up example task and parser files; docs updates

## [0.5.1] - 2025-06-13

- Make the Studio UI more readable; add advanced field options and a remove-field button
- Improve list/dict type handling in the schema builder
- Add support for importing existing parsers

## [0.5.0] - 2025-05-21

- Initial 0.5.x release baseline

[Unreleased]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.6.1...HEAD
[0.6.1]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.5.14...v0.6.0
[0.5.14]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.5.13...v0.5.14
[0.5.13]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.5.12...v0.5.13
[0.5.12]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.5.11...v0.5.12
[0.5.11]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.5.10...v0.5.11
[0.5.10]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.5.9...v0.5.10
[0.5.9]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.5.8...v0.5.9
[0.5.8]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.5.5...v0.5.8
[0.5.5]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.5.1...v0.5.5
[0.5.1]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/DIAGNijmegen/llm_extractinator/releases/tag/v0.5.0
