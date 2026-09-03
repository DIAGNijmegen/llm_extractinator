---
paths:
  - "llm_extractinator/budget.py"
  - "llm_extractinator/prediction_task.py"
  - "llm_extractinator/data_loader.py"
  - "llm_extractinator/output_parsers.py"
  - "llm_extractinator/run_config.py"
  - "llm_extractinator/main.py"
  - "tests/test_budget.py"
  - "tests/test_context_budget.py"
  - "tests/test_output_budget.py"
  - "tests/test_calibration.py"
---

# You are in the sizing path

```
prompt_tokens + num_predict <= num_ctx <= min(hardware_ceiling, model_native_max)
```

## Who decides what

| Number | Decided by | Never decided anywhere else |
|---|---|---|
| `num_ctx`, `num_predict` | `budget.resolve_budget()` | Yes |
| the generation budget's inputs | `TaskRunner._output_budget()` | Yes |
| the prompt estimate | `DataLoader.estimate_prompt_tokens()` | Yes |
| the schema's share of the answer | `output_parsers.estimate_output_tokens()` | Yes |

`resolve_budget` is **pure**. Keep it that way: no I/O, no config lookups, no
logging. That purity is what makes the invariant testable without running a
pipeline, and it is the reason this class of bug stopped recurring.

Ordering matters as much as the arithmetic. The generation budget must be final
*before* the window is sized. `PredictionTask.__init__` used to raise
`num_predict` after `TaskRunner` had already sized `num_ctx` to contain it, which
is how a generation budget eight times its own window reached the model.

## The three-way trade

- `num_ctx` is a **VRAM cost**, paid up front. Ollama allocates the KV cache on
  request and multiplies it by `OLLAMA_NUM_PARALLEL`.
- `num_predict` is a **leash, not a reservation**. Raising it does not make the
  answer better; on a thinking model it grants permission to think longer.
- The prompt estimate is **approximate by construction** (tiktoken `cl100k_base`
  against Qwen/Phi vocabularies), which is why there is a margin and why
  `_calibrate_prompt_estimate` measures the real count before spending rows.

## Rules for changes here

1. **A ceiling below what the data wants is clamped, not rejected** — and the
   caller must quantify the cost ("K of J documents exceed the R tokens that
   leaves"). Only a ceiling that cannot hold `num_predict` raises.
2. **Never trust a server-reported token count blindly.** `prompt_eval_count` is
   bounds-checked because ollama-python#271 reports it stuck at 1026 across
   57,000-character prompts.
3. **An explicit `--max_context_len` is the user's decision.** The calibration
   correction may grow a fitted window; it may never override an explicit one.
4. **`Budget.source` must describe where the number actually came from.** It
   currently reports "fitted to the data" whenever `requested_ctx is None`, even
   when the window is dominated by a user-set `--num_predict`. That is
   ticket C1; do not add new sources without the same scrutiny.
5. **Add a test to `tests/test_budget.py` for any new branch.** The suite is
   hermetic and runs in seconds; there is no excuse for an untested branch here.

## Upstream constraints you cannot design around

- Ollama has **no tokenize endpoint** (ollama/ollama#3582 closed, PR #12030
  open). `POST /api/show?verbose=true` exposes the vocabulary; the exact prompt
  count only ever arrives as `prompt_eval_count` on a real response.
- Ollama has **no separate thinking budget** (ollama/ollama#17561, open).
  `num_predict` covers thinking and answer together. Bounding the trace is
  possible only via `think: "low"|"medium"|"high"|"max"`, or by not thinking.
- Ollama **truncates the prompt silently** past `num_ctx` (ollama/ollama#14259).
