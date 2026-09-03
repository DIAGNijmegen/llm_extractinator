---
paths:
  - "llm_extractinator/gui.py"
  - "llm_extractinator/run_config.py"
  - "llm_extractinator/theme.py"
  - "llm_extractinator/schema_builder.py"
  - "tests/test_gui_smoke.py"
  - "tests/test_run_config.py"
---

# The Studio

`gui.py` **runs Streamlit at import time**, so nothing defined in it can be unit
tested. That is the entire reason `run_config.py` exists: `RunSettings` is a pure
dataclass that turns collected settings into an `extractinate` command line, and
it is where the logic belongs. Anything you can move out of `gui.py`, move.

## Streamlit rules this file depends on

- **Every widget seeds through session state**, collected in `_WIDGET_DEFAULTS`.
  Passing `value=` or `index=` alongside a `key=` Streamlit already holds makes it
  log a warning and silently ignore the default. Do not add a widget default
  inline; add it to that dict.
- A new setting needs the widget, the `_WIDGET_DEFAULTS` entry, the settings dict
  entry, and the matching `RunSettings` field and `to_command()` branch. Missing
  the last one means the Studio shows a value it never sends.

## `to_command()` emits only non-defaults

So the command shown to the user stays readable and copy-pasteable. This has a
consequence worth remembering: leaving a widget alone must mean "decide for me",
which is why `num_predict` defaulting to `None` mattered. It used to be
`int = 512`, with 512 doubling as the "do not emit the flag" sentinel — so the
Studio displayed 512, sent nothing, and the run used a schema-derived number that
was neither. One value, three meanings.

## The three token fields are confusable, and that has bitten a real user

- **"Context ceiling (tokens)"** → `max_context_cap` — the VRAM guard.
- **"Fixed context length (tokens)"** → `max_context_len` — the window, and it is
  hidden behind the *custom* option of the "Context length strategy" radio.
- **"Max output tokens"** → `num_predict` — the answer budget, reachable with one
  checkbox.

A user meaning to set the window to 20,000 set the answer budget instead, and
nothing in the interface or the log said so. Ticket C2. When adding anything
near these, say in the help text what the *other two* are not.
