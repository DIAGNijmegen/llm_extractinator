"""What the model is actually told.

Two things were wrong here and neither was visible from the output. The system
prompt asked for behaviour that grammar-constrained decoding makes impossible,
and the schema's field descriptions — the one place a task author explains what
each field *means* — never reached the model at all: they go to Ollama inside
the ``format`` parameter, which is compiled to a grammar that keeps the
structure and discards the prose.
"""

import json
import sys
from enum import Enum, IntEnum
from typing import List, Literal, Optional

import pytest
from langchain_core.example_selectors.base import BaseExampleSelector
from pydantic import BaseModel
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


def test_nested_objects_are_described():
    """Task 998's schema is a list of objects; naming only the outer field would
    leave the model to guess what each object contains."""
    task = json.loads((TASK_DIR / "Task998_example2.json").read_text("utf-8"))
    guide = describe_fields(resolve_parser_model(task, TASK_DIR))

    assert "products (list of objects)" in guide
    assert "- name (text)" in guide
    assert "- price (number)" in guide


# ── nested schemas, at every depth ────────────────────────────────


class _Cassette(BaseModel):
    tissue_type: Literal["LUNG", "BRONCHUS", "LYMPH NODE", "OTHER"] = PydanticField(
        description="Tissue in this cassette, as named in its own block"
    )
    evidence: str = PydanticField(description="The sentence the tissue type is read from")


class _Specimen(BaseModel):
    label: str = PydanticField(description="Specimen label, e.g. A or B")
    cassettes: List[_Cassette] = PydanticField(description="One per cassette")


class _ThreeLevel(BaseModel):
    """The shape of the pathology task that exposed the one-level cut-off."""

    specimens: List[_Specimen] = PydanticField(description="One per specimen")


def test_a_three_level_schema_reaches_the_model_down_to_its_leaves():
    """The guide used to stop at ``cassettes (list of objects)``.

    Every Cassette field, its description and its allowed values were cut, so
    the only tissue types the model read were the ones in the task Description.
    """
    guide = describe_fields(_ThreeLevel)

    assert "- specimens (list of objects): One per specimen" in guide
    assert "    - cassettes (list of objects): One per cassette" in guide
    assert (
        "        - tissue_type (one of: LUNG, BRONCHUS, LYMPH NODE, OTHER): "
        "Tissue in this cassette, as named in its own block"
    ) in guide
    assert (
        "        - evidence (text): The sentence the tissue type is read from"
    ) in guide


class _Node(BaseModel):
    name: str
    children: List["_Node"] = PydanticField(default_factory=list)


_Node.model_rebuild()


def test_a_self_referencing_model_terminates_and_says_so():
    guide = describe_fields(_Node)

    assert guide.splitlines() == [
        "- name (text)",
        "- children (list of objects; same shape as the top-level object)",
    ]


class _Holder(BaseModel):
    node: _Node


def test_a_recursive_model_below_the_root_points_back_at_its_own_field():
    guide = describe_fields(_Holder)

    assert "    - children (list of objects; same shape as node above)" in guide


class _A(BaseModel):
    a_value: int
    b: Optional["_B"] = None


class _B(BaseModel):
    b_value: str
    a: Optional[_A] = None


_A.model_rebuild()


def test_a_mutually_referencing_pair_terminates():
    guide = describe_fields(_A)

    assert guide.splitlines() == [
        "- a_value (whole number)",
        "- b (object; optional)",
        "    - b_value (text)",
        "    - a (object; optional; same shape as the top-level object)",
    ]


class _Pair(BaseModel):
    left: _Cassette
    right: _Cassette


def test_one_model_in_two_sibling_fields_is_described_under_both():
    """The guard is the current path, not every model seen so far."""
    guide = describe_fields(_Pair)

    assert guide.count("tissue_type (one of:") == 2
    assert "same shape as" not in guide


class _MaybeItems(BaseModel):
    items: List[Optional[_Cassette]]


def test_a_list_of_optional_models_is_followed():
    guide = describe_fields(_MaybeItems)

    assert "- items (list of objects)" in guide
    assert "    - evidence (text)" in guide


def test_a_level_three_description_is_counted_in_the_scaffolding():
    """The guide is in the scaffolding the budget counts, at every depth."""
    from llm_extractinator.data_loader import DataLoader

    class Bare(BaseModel):
        tissue_type: str

    class Described(BaseModel):
        tissue_type: str = PydanticField(
            description="Tissue in this cassette, as named in its own block "
            "of the report rather than in the summary"
        )

    def three_levels(leaf):
        class Middle(BaseModel):
            cassettes: List[leaf]

        class Top(BaseModel):
            specimens: List[Middle]

        return Top

    counter = DataLoader()
    bare, described = (
        counter.count_tokens(
            prompt_scaffolding_text(description="d", fields=describe_fields(three_levels(leaf)))
        )
        for leaf in (Bare, Described)
    )

    assert described > bare


# ── Enum classes are allowed values ───────────────────────────────


class _Tissue(str, Enum):
    LUNG = "lung"
    BRONCHUS = "bronchus"


if sys.version_info >= (3, 11):
    from enum import StrEnum

    class _TissueStrEnum(StrEnum):
        LUNG = "lung"
        BRONCHUS = "bronchus"

else:  # pragma: no cover - StrEnum is 3.11+, the package supports 3.10
    _TissueStrEnum = None


class _Laterality(Enum):
    LEFT = "left"
    RIGHT = "right"


class _Grade(IntEnum):
    LOW = 1
    HIGH = 3


@pytest.mark.parametrize(
    "enum_cls, phrase",
    [
        (_Tissue, "one of: lung, bronchus"),
        pytest.param(
            _TissueStrEnum,
            "one of: lung, bronchus",
            marks=pytest.mark.skipif(_TissueStrEnum is None, reason="StrEnum is 3.11+"),
        ),
        (_Laterality, "one of: left, right"),
        (_Grade, "one of: 1, 3"),
    ],
)
def test_an_enum_field_lists_its_values(enum_cls, phrase):
    """The grammar enforces an Enum's values; the model must be shown them.

    Values, not member names: ``LEFT`` is what the class calls it, ``left`` is
    what the grammar accepts.
    """

    class M(BaseModel):
        x: enum_cls

    assert describe_fields(M) == f"- x ({phrase})"


def test_a_list_of_enums_and_an_optional_enum_read_like_literals():
    class M(BaseModel):
        many: List[_Tissue]
        maybe: Optional[_Tissue] = None

    guide = describe_fields(M)

    assert "- many (list, each one of: lung, bronchus)" in guide
    assert "- maybe (one of: lung, bronchus; optional)" in guide


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
