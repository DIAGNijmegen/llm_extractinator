---
paths:
  - "llm_extractinator/prompt_utils.py"
  - "llm_extractinator/predictor.py"
  - "llm_extractinator/output_parsers.py"
  - "llm_extractinator/validator.py"
  - "llm_extractinator/prompt_templates/**"
  - "tests/test_prompts.py"
  - "tests/test_validator.py"
  - "tests/test_failure_reporting.py"
---

# Prompting and failure handling

## The schema does not reach the model as words

`format=<json schema>` is **grammar-constrained decoding**. llama.cpp compiles
the schema to a GBNF grammar, which keeps structure and discards prose — so
field `description` text in the pydantic model is invisible to the model.
`prompt_utils.describe_fields()` exists to render name, type-in-words, allowed
values, optionality and description into the system prompt instead. Nested
models render one level deep (`_MAX_NESTING`).

Two consequences people get wrong:

- Adding a description to a field does nothing on its own. It must go through
  `describe_fields`.
- The field guide is **prompt text and scales with the schema**, so
  `prompt_scaffolding_text()` counts all three substituted parts. Leaving it
  uncounted under-reserves the window by exactly what the guide adds, and the
  calibration probe will then refuse the run. On a wide schema the scaffolding
  can exceed the longest document — measured at 4,983 tokens against a 2,357
  token document.

## Do not instruct behaviour the grammar makes impossible

These were all real bugs:

- "Think step by step" when the first token must be `{`.
- "Return `null` or N/A" for a required int — impossible, so the model fabricated
  a number. The missing-value rule is now conditional on the field being optional.
- Forcing `required` to every property. Pydantic omits `required` only when every
  field has a default — i.e. exactly the all-optional case — so the injection
  turned "everything optional" into "everything mandatory".

`.partial()` substitution is safe: LangChain does not rescan substituted values,
so braces in a description, an example output or a document do not break the
template. Verified in all three positions.

## Retry policy

`RETRYABLE_ERRORS` is `(ConnectionError, httpx.TransportError)` — transport only.
The retry wraps **the model call alone, not the parse**, so a parse failure
structurally cannot trigger another generation whatever the type list later
becomes. Both mechanisms matter: the types state intent, the scoping enforces it.

`ollama.ResponseError` is excluded because a 400 or 404 will answer identically
next time. That reasoning does **not** hold for 503/429 on a busy shared server —
see ticket B2.

A retry is legitimate whenever the *request changes*. Re-sending with a larger
budget or a lower thinking level is a different request; re-sending the same
bytes at temperature 0 is not.

## What a failed row carries

Schema fields `None`, plus `status`, `error_type`, `error_message`, `raw_output`.
Successful rows carry the diagnostic keys as `None` so every row in a file has
one uniform key set.

Never a type default. `handle_failure` once returned `random.choice(literals)`
for a `Literal` field and `0` for an int — a failed heart-rate row became a real
zero in a mean, non-reproducibly.

`raw_output` comes from `OutputParserException.llm_output`, which is the
**content channel only**. With `reasoning=True` the trace goes to
`additional_kwargs["reasoning_content"]` and is currently discarded — which is
why a row truncated during thinking looks like an empty string. Ticket A2.

## Deleted on purpose

`prompt_templates/output_fixing/` and `validator.validate_results`, both unwired.
With `format=` bound, structurally invalid JSON is near-impossible; the realistic
failure is truncation, and a repair prompt (original instructions + broken
completion) is strictly longer than what already overflowed. Revisit only if
`failures.json` shows a category it would fix.
