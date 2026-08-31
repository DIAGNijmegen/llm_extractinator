import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Type, Union, get_args, get_origin

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)

#: Used when a task file has no ``Description``. Rendering the literal string
#: "None" into the system prompt, which is what happened before, is worse than
#: saying nothing at all.
DEFAULT_DESCRIPTION = "Extract the fields listed below from the text."

_SIMPLE_TYPE_PHRASES = {
    str: "text",
    int: "whole number",
    float: "number",
    bool: "true or false",
    list: "list",
    dict: "object",
}

#: Read better inside "list of ...".
_PLURAL_PHRASES = {
    "object": "objects",
    "text": "text values",
    "whole number": "whole numbers",
    "number": "numbers",
    "true or false": "true/false values",
}

#: How far to recurse into nested models. One level covers the realistic case —
#: a list of objects — without letting a deeply nested schema quietly triple the
#: size of every prompt.
_MAX_NESTING = 1


def _unwrap_optional(annotation):
    """Split ``Optional[X]`` into ``(X, True)``; anything else into ``(it, False)``."""
    if get_origin(annotation) is Union:
        present = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(present) < len(get_args(annotation)):
            return (present[0] if len(present) == 1 else annotation), True
    return annotation, False


def _type_phrase(annotation) -> str:
    """A field's type, in words a model can act on rather than in Python."""
    annotation, _ = _unwrap_optional(annotation)
    origin = get_origin(annotation)

    if origin is Literal:
        return "one of: " + ", ".join(str(value) for value in get_args(annotation))
    if origin in (list, List):
        args = get_args(annotation)
        if not args:
            return "list"
        inner = _type_phrase(args[0])
        if inner.startswith("one of:"):
            return f"list, each {inner}"
        return f"list of {_PLURAL_PHRASES.get(inner, inner)}"
    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            return "object"
        phrase = _SIMPLE_TYPE_PHRASES.get(annotation)
        if phrase:
            return phrase
    return "value"


def _nested_model(annotation):
    """The model inside ``X`` or ``list[X]``, if there is one."""
    annotation, _ = _unwrap_optional(annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    if get_origin(annotation) in (list, List):
        args = get_args(annotation)
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0]
    return None


def describe_fields(model: Type[BaseModel], _depth: int = 0) -> str:
    """The output schema written out for the model to read.

    The schema itself reaches Ollama as the ``format`` parameter, which compiles
    it into a grammar — and grammar compilation uses the *structure* (types,
    enums, which fields are required) while discarding every ``description``. So
    a field documented as "the primary diagnosis, verbatim from the report"
    arrived at the model as the bare key ``diagnosis``, and everything the task
    author wrote to explain it was silently dropped. Putting it in the prompt is
    the only route by which the model gets to read it.
    """
    indent = "    " * _depth
    lines = []
    for name, field in model.model_fields.items():
        annotation, optional = _unwrap_optional(field.annotation)
        # Semicolon, not comma: an enum field renders as "one of: a, b, c", and
        # appending ", optional" to that reads as though "optional" were one of
        # the allowed values.
        bits = [_type_phrase(annotation)]
        if optional:
            bits.append("optional")

        line = f"{indent}- {name} ({'; '.join(bits)})"
        if field.description:
            line += f": {field.description}"
        lines.append(line)

        nested = _nested_model(annotation)
        if nested is not None and _depth < _MAX_NESTING:
            lines.append(describe_fields(nested, _depth + 1))

    return "\n".join(lines)


def task_description(task_config: Dict[str, Any]) -> str:
    """The task's own description, or a usable stand-in."""
    description = (task_config or {}).get("Description")
    if description:
        return str(description)
    logger.warning(
        "This task file has no 'Description'; the prompt will fall back to a "
        "generic one. Describing the task is the single biggest influence on "
        "extraction quality."
    )
    return DEFAULT_DESCRIPTION


def extra_instructions(task_config: Dict[str, Any]) -> str:
    """Optional per-task text appended to the built-in system prompt.

    ``Description`` was the author's only lever, and it lands in the middle of
    instructions they can neither see nor override. This is an explicit place to
    add to them — for a house style, a language, a domain convention.
    """
    extra = (task_config or {}).get("Extra_Instructions")
    return f"\n{extra}\n" if extra else ""


def load_template(name: str) -> str:
    return (Path(__file__).parent / "prompt_templates" / f"{name}.txt").read_text()


def prompt_scaffolding_text(
    description: str = "",
    fields: str = "",
    extra_instructions: str = "",
) -> str:
    """The fixed text wrapped around every extraction prompt.

    Counting this is how the context estimate accounts for the system prompt and
    the human-message framing instead of guessing at them with a flat constant.
    Every substituted part is included, because every part is prompt text the
    run pays for — the field guide especially, which grows with the schema.

    (The schema's *structure* is still not counted: it goes to Ollama as the
    ``format`` parameter for grammar-constrained decoding, not as prompt text.
    Its descriptions are counted, because those do go into the prompt.)
    """
    system = (
        load_template("data_extraction/system_prompt")
        .replace("{description}", description or "")
        .replace("{fields}", fields or "")
        .replace("{extra_instructions}", extra_instructions or "")
    )
    return system + load_template("data_extraction/human_prompt")


def build_zero_shot_prompt() -> ChatPromptTemplate:
    system_text = load_template("data_extraction/system_prompt")
    human_text = load_template("data_extraction/human_prompt")
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("human", human_text),
        ]
    )


def build_few_shot_prompt(example_selector) -> ChatPromptTemplate:
    system_text = load_template("data_extraction/system_prompt")

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", load_template("data_extraction/human_prompt")),
            ("ai", load_template("data_extraction/ai_prompt")),
        ]
    )

    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector,
        input_variables=["input"],
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            few_shot,
            ("human", "{input}"),
        ]
    )


def build_translation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", load_template("translation/system_prompt")),
            ("human", load_template("translation/human_prompt")),
        ]
    )
