"""The context-budget invariant, tested where it is decided.

``tests/test_context_budget.py`` asserts the same relationship at the far end of
a pipeline run, which is how you find out it broke. This module asserts it on
the pure function, which is how you find out *why*.
"""

import itertools

import pytest

from llm_extractinator.budget import (
    REASONING_ALLOWANCE,
    Budget,
    ContextBudgetError,
    resolve_budget,
)


# ── fitting to the data ───────────────────────────────────────────


def test_a_fitted_window_is_exactly_the_prompt_plus_the_answer():
    budget = resolve_budget(prompt_tokens=1000, output_tokens=512)

    assert budget.num_ctx == 1512
    assert budget.num_predict == 512
    assert budget.prompt_room == 1000
    assert budget.source == "fitted to the data"
    assert budget.limited_by is None
    assert not budget.is_clamped


def test_an_explicit_window_is_honoured():
    budget = resolve_budget(prompt_tokens=1000, output_tokens=512, requested_ctx=4096)

    assert budget.num_ctx == 4096
    assert budget.source == "--max_context_len"
    assert not budget.is_clamped


# ── ceilings ──────────────────────────────────────────────────────


def test_the_hardware_cap_clamps_a_fitted_window():
    """Capping is a trade, not an error.

    Truncating a handful of outlier documents to stay inside a GPU is a
    legitimate choice when the alternative is not running at all.
    """
    budget = resolve_budget(
        prompt_tokens=10_000, output_tokens=512, hardware_ceiling=4096
    )

    assert budget.num_ctx == 4096
    assert budget.limited_by == "--max_context_cap"
    assert budget.is_clamped
    assert budget.fitted_ctx == 10_512  # what the data actually wanted


def test_the_models_native_limit_clamps_too_and_is_named():
    budget = resolve_budget(
        prompt_tokens=100_000, output_tokens=512, model_native_max=8192
    )

    assert budget.num_ctx == 8192
    assert "native context" in budget.limited_by


def test_the_tighter_of_two_ceilings_wins():
    tight_card = resolve_budget(
        prompt_tokens=99_000, output_tokens=512,
        hardware_ceiling=4096, model_native_max=32_768,
    )
    tight_model = resolve_budget(
        prompt_tokens=99_000, output_tokens=512,
        hardware_ceiling=32_768, model_native_max=4096,
    )

    assert tight_card.num_ctx == tight_model.num_ctx == 4096
    assert tight_card.limited_by == "--max_context_cap"
    assert "native context" in tight_model.limited_by


def test_a_ceiling_above_the_fitted_window_changes_nothing():
    budget = resolve_budget(
        prompt_tokens=1000, output_tokens=512,
        hardware_ceiling=1_000_000, model_native_max=1_000_000,
    )
    assert budget.num_ctx == 1512
    assert budget.limited_by is None


# ── impossible configurations fail, naming the term ───────────────


def test_a_window_that_cannot_hold_the_answer_is_refused():
    with pytest.raises(ContextBudgetError, match="must be smaller than num_ctx"):
        resolve_budget(prompt_tokens=100, output_tokens=512, requested_ctx=256)


def test_the_message_names_the_ceiling_that_caused_it():
    """"Too small" is not actionable unless you know which knob made it small."""
    with pytest.raises(ContextBudgetError, match="--max_context_cap"):
        resolve_budget(
            prompt_tokens=100, output_tokens=5512, hardware_ceiling=4096
        )

    with pytest.raises(ContextBudgetError, match="native context"):
        resolve_budget(
            prompt_tokens=100, output_tokens=5512, model_native_max=2048
        )

    with pytest.raises(ContextBudgetError, match="--max_context_len"):
        resolve_budget(prompt_tokens=100, output_tokens=5512, requested_ctx=2048)


def test_a_non_positive_generation_budget_is_refused():
    """-1 (unbounded) and -2 (fill the context) are both bad defaults for
    extraction — a looping model runs to the context limit."""
    for bad in (0, -1, -2):
        with pytest.raises(ContextBudgetError, match="must be positive"):
            resolve_budget(prompt_tokens=100, output_tokens=bad)


def test_a_non_positive_window_is_refused():
    with pytest.raises(ContextBudgetError, match="must be positive"):
        resolve_budget(prompt_tokens=100, output_tokens=512, requested_ctx=0)


# ── the invariant itself ──────────────────────────────────────────


def test_the_invariant_holds_across_every_combination_that_resolves():
    """Whatever comes out, it satisfies the relationship — or it raised.

    The failure mode being guarded against is a resolver that returns numbers
    which look plausible but do not add up; sweeping the space is cheaper than
    reasoning about which branch might do that.
    """
    values = [1, 100, 512, 5512, 4096, 100_000]
    ceilings = [None, 512, 4096, 32_768]

    resolved = 0
    for prompt, output, requested, cap, native in itertools.product(
        values, [512, 5512], [None, 2048, 32_768], ceilings, ceilings
    ):
        try:
            budget = resolve_budget(
                prompt_tokens=prompt, output_tokens=output,
                requested_ctx=requested, hardware_ceiling=cap,
                model_native_max=native,
            )
        except ContextBudgetError:
            continue
        resolved += 1

        assert budget.num_predict < budget.num_ctx
        assert budget.prompt_tokens + budget.num_predict <= budget.num_ctx
        assert budget.prompt_room > 0
        for ceiling in (cap, native):
            if ceiling is not None:
                assert budget.num_ctx <= ceiling

    assert resolved > 100, f"only {resolved} combinations resolved; sweep too narrow"


def test_a_clamped_window_reserves_less_for_the_prompt():
    """The reserved prompt has to shrink with the window, or the Budget itself
    would carry numbers that violate the invariant it exists to enforce."""
    budget = resolve_budget(
        prompt_tokens=10_000, output_tokens=512, hardware_ceiling=4096
    )
    assert budget.prompt_tokens == 4096 - 512
    assert budget.prompt_tokens + budget.num_predict == budget.num_ctx


# ── the log line ──────────────────────────────────────────────────


def test_describe_shows_where_the_window_went():
    plain = resolve_budget(prompt_tokens=1119, output_tokens=512, requested_ctx=4096)
    assert plain.describe() == (
        "context 4096 = 1119 prompt + 512 answer + 2465 headroom "
        "(--max_context_len)"
    )

    capped = resolve_budget(
        prompt_tokens=10_000, output_tokens=512, hardware_ceiling=4096
    )
    assert "capped by --max_context_cap" in capped.describe()


def test_reasoning_allowance_is_a_named_constant():
    """It is referenced from the runner and from the docs; a bare 5000 in three
    places is how it previously drifted out of step with the window."""
    assert REASONING_ALLOWANCE == 5000
