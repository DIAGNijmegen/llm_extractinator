from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from llm_extractinator.predictor import Predictor

model = ChatOllama(
    model="qwen2.5:0.5b",
    temperature=0.0,
    num_predict=512,
    num_ctx=1024,
    format="json",
    seed=42,
)

# Mock Example Path
examples_path = Path("examples.json")

# Create Predictor instance
predictor = Predictor(
    model=model,
    task_config=task_config,
    examples_path=None,
    num_examples=0,
)


# Mock Parser Model
class MockParserModel(BaseModel):
    translation: str
