---
paths:
  - ".github/workflows/**"
  - "pyproject.toml"
  - "MANIFEST.in"
  - "Dockerfile"
  - ".dockerignore"
  - "build.sh"
  - "build.ps1"
  - "mkdocs.yml"
  - "llm_extractinator/__init__.py"
  - "CHANGELOG.md"
---

# You are in the release path

**Merging to `main` publishes to PyPI.** `.github/workflows/publish.yml` fires on
push to `main` and on a created release, builds, and uploads with
`PYPI_API_TOKEN`. Nothing else guards it — no manual approval, no tag check, no
version-already-exists check. `main` is a release, not a staging area, and a
merge is a publication. Treat any change here accordingly.

## The facts, verified 2026-09-03

- **The version lives in `llm_extractinator/__init__.py` and nowhere else.**
  `pyproject.toml` reads it via `[tool.setuptools.dynamic] version = { attr = ... }`.
  Editing the version in two places is how you get a wheel that disagrees with
  itself.
- **Branches are named for the version** they carry (`0.7.0`, `0.8.0`).
- **`CHANGELOG.md`'s working heading is `## [Unreleased]`**, promoted to
  `## [x.y.z] - <date>` at release. Entries are written from the user's point of
  view — what changed for someone running the tool, not which function moved.
- **`test.yml`** runs on push to any branch and on PRs to `main`: Python 3.11,
  `pip install -e ".[test]"`, `pytest -m "not integration"`. Integration tests
  are excluded because they need Ollama and a GPU.
- **`MANIFEST.in`** ships `README.md`, `LICENSE`, `pyproject.toml` and
  `recursive-include llm_extractinator`. Root markdown — `CLAUDE.md`,
  `PLAN_*.md` — does **not** reach the sdist. Keep it that way.
- **`.dockerignore`** excludes `*.md`, `docs/`, `.claude/`, and the runtime-mounted
  `data/`, `output/`, `tasks/`, `examples/`, `ollama_models/`.
- **There is no formatter or linter in CI.** `pytest` is the only gate. If you
  are adding one, read the note below first.

## Before changing anything here

`pyproject.toml` and `.github/**` are on the **ask** list in
`.claude/settings.json` deliberately. A change here can publish a package or
change what ships inside one. Do not treat it as ordinary config editing, and do
not batch it into a functional commit — release-mechanics changes get their own
commit so they can be reverted alone.

## The formatter question, and why it is not a quick win

`black --check` on 2026-09-03: **22 of 42 files would be reformatted.** So
introducing a formatter is not a config addition, it is a bulk rewrite of half
the tree. Consequences that decide the timing:

- It collides with **everything in flight**. A 22-file reformat landing beside
  Lane A means every logical change arrives tangled in whitespace.
- It destroys `git blame` on the files being actively worked, which in this
  repo is where the reasoning lives — the docstrings are the design record.

**Therefore: after 0.8.0 ships, not during.** When it is done, do it as one
commit that touches nothing else, add a `.git-blame-ignore-revs` naming that
commit (and `blame.ignoreRevsFile` in the repo config) so history stays
readable, and add the `--check` to `test.yml` and the `PostToolUse` hook to
`.claude/settings.json` in the *same* commit — a formatter that is not enforced
drifts back within a release.

Adding the hook *before* the bulk reformat is the worst of both: agent-touched
files get reformatted piecemeal, so reformat noise arrives scattered through
functional diffs instead of isolated in one commit.
