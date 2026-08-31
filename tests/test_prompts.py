"""What the model is actually told.

Two things were wrong here and neither was visible from the output. The system
prompt asked for behaviour that grammar-constrained decoding makes impossible,
and the schema's field descriptions — the one place a task author explains what
each field *means* — never reached the model at all: they go to Ollama inside
the ``format`` parameter, which is compiled to a grammar that keeps the
structure and discards the prose.
"""

import json

from langchain_core.example_selectors.base import BaseExampleSelector
from pydantic import Field as PydanticField

from llm_extractinator.output_parsers import load_parser, resolve_parser_model
from llm_extractinator.prompt_utils import (
    DEFAULT_DESCRIPTION,
    build_few_shot_prompt,
    build_zero_shot_prompt,
    describe_fields,
    extra_instructions,
    prompt_scaffolding_text,
    task_description,
)
from tests.conftest import TASK_DIR, FakeChatModel, load_predictions

HR_RESPONSE = '{"HR": 78, "Name": "Alice"}'

RICH_SCHEMA = {
    "diagnosis": {"type": "str", "description": "Primary diagnosis, verbatim"},
    "severity": {
        "type": "str",
        "description": "Severity as stated",
        "literals": ["mild", "moderate", "severe"],
    },
    "lesion_count": {"type": "int", "description": "Number of distinct lesions"},
    "follow_up": {
        "type": "str",
        "description": "Recommended follow-up",
        "optional": True,
    },
}


def _rendered(task_config, model, input_text="a report"):
    return build_zero_shot_prompt().partial(
        description=task_description(task_config),
        fields=describe_fields(model),
        extra_instructions=extra_instructions(task_config),
    ).format_messages(input=input_text)[0].content


# ── the schema reaches the model ──────────────────────────────────


def test_field_descriptions_appear_in_the_prompt():
    """The whole point. Without this the model sees only the key name."""
    model = load_parser("Extraction", RICH_SCHEMA)
    guide = describe_fields(model)

    assert "Primary diagnosis, verbatim" in guide
    assert "Number of distinct lesions" in guide


def test_allowed_values_are_listed():
    guide = describe_fields(load_parser("Extraction", RICH_SCHEMA))
    assert "one of: mild, moderate, severe" in guide


def test_optional_is_not_mistaken_for_an_allowed_value():
    """"one of: mild, moderate, severe, optional" reads as four choices.

    A semicolon separates the marker from the enum, so it cannot be misread as
    one more permitted value.
    """
    optional_enum = {
        "severity": {
            "type": "str",
            "description": "Severity if stated",
            "literals": ["mild", "moderate", "severe"],
            "optional": True,
        }
    }
    guide = describe_fields(load_parser("Extraction", optional_enum))

    assert "one of: mild, moderate, severe; optional" in guide
    assert "severe, optional" not in guide


def test_types_are_described_in_words_not_python():
    guide = describe_fields(load_parser("Extraction", RICH_SCHEMA))
    assert "whole number" in guide
    assert "class 'int'" not in guide
    assert "typing." not in guide


def test_nested_objects_are_described_one_level_deep():
    """Task 998's schema is a list of objects; naming only the outer field would
    leave the model to guess what each object contains."""
    task = json.loads((TASK_DIR / "Task998_example2.json").read_text("utf-8"))
    guide = describe_fields(resolve_parser_model(task, TASK_DIR))

    assert "products (list of objects)" in guide
    assert "- name (text)" in guide
    assert "- price (number)" in guide


# ── what the prompt no longer says ────────────────────────────────


def test_the_prompt_does_not_ask_for_impossible_reasoning():
    """Every request binds ``format=<schema>``, so the first token must be ``{``.

    "Think step by step" instructed behaviour the decoder makes impossible: at
    best dead tokens, at worst probability pushed toward a prose opening the
    grammar then overrides.
    """
    model = load_parser("Extraction", RICH_SCHEMA)
    rendered = _rendered({"Description": "d"}, model)

    assert "step by step" not in rendered.lower()


def test_the_prompt_does_not_contradict_the_schema_about_missing_values():
    """The old text said to return `null` or "N/A" for anything missing.

    For a required integer the grammar permits neither, so the model had to emit
    *something* — a fabricated number, which is the one output an extraction
    tool must not produce. The rule is now conditional on the field actually
    being optional.
    """
    model = load_parser("Extraction", RICH_SCHEMA)
    rendered = _rendered({"Description": "d"}, model)

    assert "N/A" not in rendered
    assert "marked optional" in rendered


# ── user-supplied text ────────────────────────────────────────────


def test_a_task_without_a_description_does_not_say_None():
    """``task_config.get("Description")`` returned None, which rendered as the
    literal string "None" in the middle of the system prompt."""
    model = load_parser("Extraction", RICH_SCHEMA)
    rendered = _rendered({}, model)

    assert "Task: None" not in rendered
    assert DEFAULT_DESCRIPTION in rendered


def test_extra_instructions_are_appended_when_a_task_supplies_them():
    """The description was the author's only lever, and it lands mid-template
    among instructions they can neither see nor override."""
    model = load_parser("Extraction", RICH_SCHEMA)
    rendered = _rendered(
        {"Description": "d", "Extra_Instructions": "Answer in Dutch."}, model
    )
    assert "Answer in Dutch." in rendered


def test_extra_instructions_are_absent_when_not_supplied():
    assert extra_instructions({"Description": "d"}) == ""


def test_braces_in_user_supplied_text_are_not_treated_as_placeholders():
    """Descriptions, field descriptions and documents are all user text.

    LangChain substitutes partial *values* without rescanning them, so braces
    survive — but this is now three sources of user text rather than one, and a
    regression here would be a crash at run time on somebody's real task file.
    """
    schema = {"patient_id": {"type": "str", "description": "The {id} in braces {1,2}"}}
    model = load_parser("Extraction", schema)
    rendered = _rendered(
        {"Description": "Find the {patient_id} and sets like {1,2,3}."},
        model,
        input_text="Patient {x}.",
    )

    assert "{patient_id}" in rendered
    assert "{id} in braces {1,2}" in rendered


# ── the template and its callers stay in step ─────────────────────


def test_every_template_placeholder_is_supplied_by_the_caller():
    """A placeholder added to the template but not wired up is a run-time crash.

    ``input`` is the per-row variable; everything else has to be partialled in
    by ``Predictor.prepare_prompt_ollama`` before the template is ever used.
    """
    for template in (
        build_zero_shot_prompt(),
        build_few_shot_prompt(example_selector=_StaticSelector()),
    ):
        assert set(template.input_variables) - {"input"} == {
            "description",
            "fields",
            "extra_instructions",
        }


def test_the_context_estimate_counts_the_field_guide():
    """The guide is prompt text, and it scales with the schema.

    Leaving it out of the estimate would under-reserve the window by exactly the
    amount this change added — and since phase 4 the calibration probe would
    then refuse the run.
    """
    scaffolding = prompt_scaffolding_text(
        description="d", fields="- diagnosis (text): Primary diagnosis, verbatim"
    )

    assert "Primary diagnosis, verbatim" in scaffolding
    for placeholder in ("{description}", "{fields}", "{extra_instructions}"):
        assert placeholder not in scaffolding


# ── declared optionality survives to the model ────────────────────


class _StaticSelector(BaseExampleSelector):
    """A selector stand-in, for template introspection only."""

    def select_examples(self, input_variables):  # pragma: no cover
        return []

    def add_example(self, example):  # pragma: no cover
        pass


class _BindRecordingModel(FakeChatModel):
    """Records the ``format`` schema the pipeline binds to the model."""

    bound: dict = PydanticField(default_factory=dict)

    def bind(self, **kwargs):
        self.bound.update(kwargs)
        return super().bind(**kwargs)


def test_an_all_optional_schema_is_not_forced_to_be_mandatory(offline_run):
    """``predict`` used to set ``required`` to every property when pydantic
    omitted the key — and pydantic omits it only when every field has a default,
    which is exactly when the author marked them all optional. The model was
    then obliged to invent values for fields it had permission to omit.
    """
    model = _BindRecordingModel(responses=["{}"])
    offline_run(model=model, task_id=997, run_name="opt")

    schema = model.bound["format"]
    assert "required" not in schema or schema["required"] == [], (
        f"declared-optional fields were forced to be required: "
        f"{schema.get('required')}"
    )


def test_a_schema_with_required_fields_keeps_them_required(offline_run):
    """The other direction: removing the injection must not lose real
    requirements, which pydantic emits by itself."""
    model = _BindRecordingModel(responses=[HR_RESPONSE])
    offline_run(model=model, task_id=999, run_name="req")

    assert set(model.bound["format"]["required"]) == {"HR", "Name"}


def test_an_inline_schema_task_runs_end_to_end(offline_run):
    """Task 997 declares its schema inline rather than in a parser file; that
    branch of ``resolve_parser_model`` had no coverage at all."""
    out = offline_run(responses=['{"diagnosis": "none"}'], task_id=997, run_name="inline")
    preds = load_predictions(out, "inline", "Task997_optional")

    assert len(preds) == 5
    assert all(row["status"] == "success" for row in preds)
