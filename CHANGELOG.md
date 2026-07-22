# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/DIAGNijmegen/llm_extractinator/compare/v0.6.0...HEAD
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
