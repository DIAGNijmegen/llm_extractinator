import importlib.util
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field, create_model
from pydantic.types import StrictBool

logger = logging.getLogger(__name__)

type_mapping = {
    "str": str,
    "int": int,
    "float": float,
    "bool": StrictBool,
    "list": list,
    "dict": dict,
    "any": Any,
}


def create_field(field_info: Dict[str, Any], parent_name: str) -> Tuple[Any, Field]:
    """
    Create a Pydantic field with a type, description, and optional Literal values.
    """
    field_type = type_mapping.get(field_info["type"], None)
    description = field_info.get("description", None)
    is_optional = field_info.get("optional", False)

    # Handle nested dict objects
    if field_info["type"] == "dict":
        model_name = f"{parent_name}_{field_info.get('name', 'Dict')}"
        nested_model = create_pydantic_model_from_json(
            field_info["properties"], model_name=model_name
        )
        field_type = nested_model

    # Handle list types
    elif field_info["type"] == "list":
        item_type_info = field_info.get("items")
        if not item_type_info:
            raise ValueError("'items' must be defined for list type fields.")

        item_type, _ = create_field(item_type_info, parent_name + "Item")
        field_type = List[item_type]

    elif field_type is None:
        raise ValueError(f"Unsupported field type: {field_info['type']}")

    # Handle literals if specified
    literals = field_info.get("literals")
    if literals:
        field_type = Literal[tuple(literals)]

    if is_optional:
        field_type = Optional[field_type]

    return field_type, Field(
        default=None if is_optional else ..., description=description
    )


def create_pydantic_model_from_json(
    data: Dict[str, Any], model_name: str = "OutputParser"
) -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model from a dictionary specification.
    """
    fields = {}

    for key, field_info in data.items():
        field_type, pydantic_field = create_field(field_info, parent_name=model_name)
        fields[key] = (field_type, pydantic_field)

    return create_model(model_name, **fields)


def load_parser(
    task_type: str, parser_format: Optional[Dict[str, Any]]
) -> Type[BaseModel]:
    """
    Load a predefined Pydantic model based on task type, or generate one dynamically.
    """
    predefined_models = {
        "Example Generation": lambda: create_model(
            "ExampleGenerationOutput",
            reasoning=(
                str,
                Field(description="The thought process leading to the answer"),
            ),
        ),
        "Translation": lambda: create_model(
            "TranslationOutput",
            translation=(str, Field(description="The text translated to English")),
        ),
    }

    if task_type in predefined_models:
        return predefined_models[task_type]()

    if parser_format is None:
        raise ValueError("parser_format must be provided for custom task types.")

    return create_pydantic_model_from_json(parser_format)


def load_parser_pydantic(parser_path: Path) -> BaseModel:
    """
    Load a Pydantic model from a python file
    """
    if parser_path.exists():
        module_name = parser_path.stem  # Get the filename without .py
        spec = importlib.util.spec_from_file_location(module_name, str(parser_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if hasattr(module, "OutputParser"):
            model_class = getattr(module, "OutputParser")
            if issubclass(model_class, BaseModel):
                return model_class
            else:
                raise TypeError(
                    f"'OutputParser' in {parser_path} is not a subclass of BaseModel"
                )
        else:
            raise ImportError(f"No OutputParser class found in {parser_path}")

    else:
        raise FileNotFoundError(f"Parser file not found in {parser_path}.")


def resolve_parser_model(
    task_config: Dict[str, Any], task_dir: Path
) -> Type[BaseModel]:
    """The output model a task describes, however it describes it.

    ``Parser_Format`` is either an inline schema dict or the filename of a Python
    module under ``<task_dir>/parsers``. Both the prompt builder and the context
    estimator need this model and they run at different points in the pipeline,
    so resolving it lives here rather than inside either of them.
    """
    parser_format = (task_config or {}).get("Parser_Format")

    if isinstance(parser_format, dict):
        try:
            return load_parser(task_type="Extraction", parser_format=parser_format)
        except KeyError as error:
            logger.error("Missing required key in parser format dictionary: %s", error)
            raise ValueError(f"Invalid parser format dictionary: {error}") from error
        except Exception as error:
            logger.error(
                "Failed to load parser model from dictionary format: %s: %s",
                type(error).__name__,
                error,
            )
            raise

    parser_path = Path(task_dir) / "parsers" / str(parser_format)
    try:
        if not parser_path.exists():
            raise FileNotFoundError(f"Parser file not found: {parser_path}")
        return load_parser_pydantic(parser_path=parser_path)
    except FileNotFoundError as error:
        logger.error(str(error))
        raise
    except (ImportError, AttributeError, SyntaxError) as error:
        logger.error(
            "Failed to import parser from %s: %s: %s",
            parser_path,
            type(error).__name__,
            error,
        )
        raise ValueError(
            f"Parser file '{parser_format}' has invalid Python code or structure"
        ) from error
    except Exception as error:
        logger.error(
            "Failed to load parser model from %s: %s: %s",
            parser_path,
            type(error).__name__,
            error,
        )
        raise


#: Rough token cost of one JSON value, by kind. Deliberately generous: an
#: under-estimate truncates the answer and costs the whole row, whereas an
#: over-estimate costs some context window. See model_sizing.md — output length
#: scales with *value length*, not field count, so free text dominates.
_VALUE_TOKENS = {
    "enum": 6,
    "bool": 2,
    "int": 4,
    "float": 6,
    "text": 30,
    "unknown": 12,
}

#: A list field has no declared length. Five is a guess, and the doubling below
#: is what covers it being wrong.
_ASSUMED_LIST_ITEMS = 5

#: `"field_name": ` — the key, its quotes, the colon and the separator.
_KEY_TOKENS = 4


def allowed_values(annotation) -> Optional[tuple]:
    """The values a field is constrained to, or ``None`` if it is open.

    A ``Literal`` and an ``Enum`` subclass are the same thing to the grammar —
    Ollama compiles both into a fixed set of permitted values — so they have to
    be the same thing everywhere else too. Before this helper existed only
    ``Literal`` was recognised: an ``Enum`` field was enforced by the grammar
    yet never listed in the prompt, so the model was held to a list it had not
    been shown, and the output estimate costed it as an unknown value.

    For an ``Enum`` these are the member *values*, not their names, because the
    values are what the JSON schema — and so the grammar — accepts.

    Both the field guide and the output estimate call this, so they cannot
    disagree about what counts as a fixed set again.
    """
    if get_origin(annotation) is Literal:
        return get_args(annotation)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return tuple(member.value for member in annotation)
    return None


def _value_tokens(annotation, _path: Tuple[type, ...] = ()) -> int:
    """Cost of one value of this annotation, following nested models.

    ``_path`` holds the models already being costed on the way down, so a
    self-referencing schema (``Node.children: list[Node]``) terminates instead
    of recursing until Python gives up. A model met again on its own path is
    costed as one value of unknown shape: a tree's depth is not in the schema,
    so any fixed number here is a guess, and this one is the cheapest that does
    not pretend otherwise.
    """
    if get_origin(annotation) is Union:
        present = [arg for arg in get_args(annotation) if arg is not type(None)]
        if present:
            annotation = present[0]

    if allowed_values(annotation) is not None:
        return _VALUE_TOKENS["enum"]
    if get_origin(annotation) in (list, List):
        args = get_args(annotation)
        inner = _value_tokens(args[0], _path) if args else _VALUE_TOKENS["unknown"]
        return _ASSUMED_LIST_ITEMS * inner

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            if annotation in _path:
                return _VALUE_TOKENS["unknown"]
            path = _path + (annotation,)
            return sum(
                _KEY_TOKENS + _value_tokens(field.annotation, path)
                for field in annotation.model_fields.values()
            )
        if annotation is bool:
            return _VALUE_TOKENS["bool"]
        if annotation is int:
            return _VALUE_TOKENS["int"]
        if annotation is float:
            return _VALUE_TOKENS["float"]
        if annotation is str:
            return _VALUE_TOKENS["text"]
    return _VALUE_TOKENS["unknown"]


def estimate_output_tokens(model: Type[BaseModel]) -> int:
    """A plausible size for the JSON answer this schema describes.

    Doubled, because every term here is a guess and the failure is asymmetric:
    too small truncates the object mid-write and loses the row, too large costs
    context window that the resolver will account for anyway.

    This does not include the reasoning allowance — a thinking model spends its
    chain of thought from the same budget, and that is added separately by
    ``TaskRunner._output_budget`` so it applies whatever the schema looks like.
    """
    body = sum(
        _KEY_TOKENS + _value_tokens(field.annotation, (model,))
        for field in model.model_fields.values()
    )
    return 2 * body
