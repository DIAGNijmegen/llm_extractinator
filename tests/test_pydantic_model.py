from typing import List, Optional, Type

import pytest
from pydantic import BaseModel, ValidationError

from llm_extractinator.output_parsers import (
    create_pydantic_model_from_json,
    load_parser,
)


def test_basic_types():
    schema = {
        "name": {"type": "str", "description": "User name"},
        "age": {"type": "int", "description": "User age"},
        "active": {"type": "bool", "description": "Is the user active?"},
    }

    Model = create_pydantic_model_from_json(schema)

    instance = Model(name="Alice", age=30, active=True)
    assert instance.name == "Alice"
    assert instance.age == 30
    assert instance.active is True

    with pytest.raises(ValidationError):
        Model(name="Alice", age="thirty", active=True)  # Invalid age type


def test_optional_fields():
    schema = {"email": {"type": "str", "description": "User email", "optional": True}}

    Model = create_pydantic_model_from_json(schema)

    instance = Model()
    assert instance.email is None  # Optional field should default to None

    instance = Model(email="user@example.com")
    assert instance.email == "user@example.com"


def test_literals():
    schema = {
        "status": {
            "type": "str",
            "description": "Account status",
            "literals": ["active", "inactive", "banned"],
        }
    }

    Model = create_pydantic_model_from_json(schema)

    instance = Model(status="active")
    assert instance.status == "active"

    with pytest.raises(ValidationError):
        Model(status="pending")  # Invalid literal value


def test_nested_dict():
    schema = {
        "profile": {
            "type": "dict",
            "description": "User profile",
            "properties": {
                "bio": {"type": "str", "description": "User biography"},
                "age": {"type": "int", "description": "User age"},
            },
        }
    }

    Model = create_pydantic_model_from_json(schema)

    instance = Model(profile={"bio": "Software engineer", "age": 25})
    assert instance.profile.bio == "Software engineer"
    assert instance.profile.age == 25

    with pytest.raises(ValidationError):
        Model(
            profile={"bio": "Software engineer", "age": "twenty-five"}
        )  # Invalid type


def test_list_of_dicts():
    schema = {
        "posts": {
            "type": "list",
            "description": "List of posts",
            "items": {
                "type": "dict",
                "properties": {
                    "title": {"type": "str", "description": "Post title"},
                    "content": {"type": "str", "description": "Post content"},
                },
            },
        }
    }

    Model = create_pydantic_model_from_json(schema)

    instance = Model(posts=[{"title": "Hello", "content": "World"}])
    assert len(instance.posts) == 1
    assert instance.posts[0].title == "Hello"

    with pytest.raises(ValidationError):
        Model(posts=[{"title": "Hello", "content": 123}])  # Invalid content type


def test_predefined_models():
    ExampleGenModel = load_parser("Example Generation", None)
    instance = ExampleGenModel(reasoning="Logical explanation")
    assert instance.reasoning == "Logical explanation"

    TranslationModel = load_parser("Translation", None)
    instance = TranslationModel(translation="Hola")
    assert instance.translation == "Hola"


def test_custom_parser():
    schema = {
        "answer": {"type": "str", "description": "Generated answer"},
        "confidence": {"type": "float", "description": "Confidence score"},
    }

    Model = load_parser("Custom Task", schema)
    instance = Model(answer="42", confidence=0.95)

    assert instance.answer == "42"
    assert instance.confidence == 0.95

    with pytest.raises(ValidationError):
        Model(answer="42", confidence="high")  # Invalid float value
