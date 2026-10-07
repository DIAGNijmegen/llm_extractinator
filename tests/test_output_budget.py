"""Sizing the answer from the schema instead of from a number the user invented.

``--num_predict`` had one value for every task — 512 — whether the schema had
three enum fields or thirty free-text ones. Too small truncates the JSON
mid-object and loses the whole row; too large costs context window. Neither is
something a user should have to guess at, so an unset value is now derived,
following the same ``None = auto`` convention as ``--temperature`` and the
reasoning flags.
"""

import json
from enum import Enum
from typing import List, Literal

import pytest
from pydantic import BaseModel

from llm_extractinator.output_parsers import (
    estimate_output_tokens,
    load_parser,
    load_parser_pydantic,
    resolve_parser_model,
)
from llm_extractinator.run_config import DEFAULT_NUM_PREDICT, suggested_num_predict
from tests.conftest import TASK_DIR

PRODUCTS_RESPONSE = '{"products": [{"name": "Widget", "price": 9.99}]}'
HR_RESPONSE = '{"HR": 78, "Name": "Alice"}'

TINY = {"flag": {"type": "bool", "description": "yes or no"}}
WIDE = {f"f{i}": {"type": "str", "description": "free text"} for i in range(30)}


# ── the estimate ──────────────────────────────────────────────────


def test_a_wider_schema_needs_a_bigger_answer():
    assert estimate_output_tokens(load_parser("Extraction", WIDE)) > estimate_output_tokens(
        load_parser("Extraction", TINY)
    )


def test_free_text_costs_more_than_a_fixed_set_of_values():
    """Output length scales with value length, not field count.

    One free-text ``summary`` can outweigh twenty enums, which is exactly why a
    flat number could not serve both.
    """
    text = load_parser("Extraction", {"x": {"type": "str", "description": "d"}})
    enum = load_parser(
        "Extraction", {"x": {"type": "str", "description": "d", "literals": ["a", "b"]}}
    )
    assert estimate_output_tokens(text) > estimate_output_tokens(enum)


def test_a_list_of_objects_accounts_for_several_items():
    """A list field has no declared length, so it cannot cost one item."""
    model = load_parser_pydantic(TASK_DIR / "parsers" / "output_parser2.py")
    one_object = estimate_output_tokens(
        load_parser(
            "Extraction",
            {"name": {"type": "str", "description": "d"},
             "price": {"type": "float", "description": "d"}},
        )
    )
    assert estimate_output_tokens(model) > one_object


def test_an_enum_field_costs_the_same_as_the_equivalent_literal():
    """Both compile to the same fixed set in the grammar.

    An Enum used to fall through to "unknown" and cost twice as much as the
    Literal with the same values.
    """

    class Tissue(str, Enum):
        LUNG = "lung"
        BRONCHUS = "bronchus"

    class WithEnum(BaseModel):
        tissue: Tissue
        many: List[Tissue]

    class WithLiteral(BaseModel):
        tissue: Literal["lung", "bronchus"]
        many: List[Literal["lung", "bronchus"]]

    assert estimate_output_tokens(WithEnum) == estimate_output_tokens(WithLiteral)


def test_a_self_referencing_schema_has_a_finite_estimate():
    """``Node.children: list[Node]`` used to recurse until RecursionError."""

    class Node(BaseModel):
        name: str
        children: List["Node"] = []

    Node.model_rebuild()

    assert 0 < estimate_output_tokens(Node) < 10_000


# ── the resolver ──────────────────────────────────────────────────


def test_a_small_schema_keeps_the_established_default():
    """Auto-sizing may raise the budget but must never lower it.

    512 was the shipped default; anything below it would be a silent regression
    on tasks that work today.
    """
    assert suggested_num_predict(parser_model=load_parser("Extraction", TINY)) == (
        DEFAULT_NUM_PREDICT
    )


def test_a_wide_schema_raises_the_budget_above_the_default():
    assert suggested_num_predict(
        parser_model=load_parser("Extraction", WIDE)
    ) > DEFAULT_NUM_PREDICT


def test_without_a_schema_it_is_the_plain_default():
    assert suggested_num_predict() == DEFAULT_NUM_PREDICT


def test_reasoning_adds_the_chain_of_thought_allowance():
    """The Studio asks for the whole figure to warn on a hand-set value; the
    runner asks without it and adds the allowance itself, so neither
    double-counts."""
    assert suggested_num_predict(reasoning=True) > suggested_num_predict(reasoning=False)
    assert suggested_num_predict(reasoning=False) == DEFAULT_NUM_PREDICT


# ── end to end ────────────────────────────────────────────────────


def test_an_unset_budget_is_sized_from_the_schema(record_model_kwargs):
    """Task 996 has twenty free-text fields — 512 would truncate it."""
    records = record_model_kwargs(
        responses=["{}"], task_id=996, run_name="wide",
        num_predict=None, max_context_len="max",
    )

    task = json.loads((TASK_DIR / "Task996_wide.json").read_text("utf-8"))
    expected = suggested_num_predict(
        parser_model=resolve_parser_model(task, TASK_DIR)
    )
    assert records[0]["num_predict"] == expected > DEFAULT_NUM_PREDICT


def test_a_narrow_schema_still_gets_the_default(record_model_kwargs):
    records = record_model_kwargs(
        responses=[HR_RESPONSE], task_id=999, run_name="narrow",
        num_predict=None, max_context_len="max",
    )
    assert records[0]["num_predict"] == DEFAULT_NUM_PREDICT


def test_an_explicit_budget_is_still_honoured(record_model_kwargs):
    """It becomes an override, not a requirement — but it is still an override."""
    records = record_model_kwargs(
        responses=["{}"], task_id=996, run_name="explicit",
        num_predict=99, max_context_len="max",
    )
    assert records[0]["num_predict"] == 99


def test_the_window_is_sized_around_the_derived_budget(record_model_kwargs):
    """Deriving the budget after the window was sized would be the original bug
    in a new place, so the invariant is asserted here too."""
    records = record_model_kwargs(
        responses=["{}"], task_id=996, run_name="wide_ctx",
        num_predict=None, max_context_len="max",
    )
    record = records[0]
    assert record["prompt_tokens"] + record["num_predict"] <= record["num_ctx"]


def test_a_thinking_model_gets_the_allowance_on_top_of_the_derived_budget(
    record_model_kwargs,
):
    plain = record_model_kwargs(
        responses=["{}"], task_id=996, run_name="wide_plain",
        num_predict=None, max_context_len="max", thinking=False,
    )
    thinking = record_model_kwargs(
        responses=["{}"], task_id=996, run_name="wide_think",
        num_predict=None, max_context_len="max", thinking=True,
    )
    assert thinking[0]["num_predict"] > plain[0]["num_predict"] > DEFAULT_NUM_PREDICT


def test_an_unreadable_schema_falls_back_rather_than_failing_the_run(
    record_model_kwargs, monkeypatch
):
    """Auto-sizing is a convenience. Failing a run over it would be worse than
    the flat number it replaces."""
    from llm_extractinator import main

    def _explode(*args, **kwargs):
        raise ValueError("schema is broken")

    monkeypatch.setattr(main, "resolve_parser_model", _explode)
    records = record_model_kwargs(
        responses=[HR_RESPONSE], task_id=999, run_name="broken",
        num_predict=None, max_context_len=4096,
    )
    assert records[0]["num_predict"] == DEFAULT_NUM_PREDICT
