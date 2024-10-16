from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator
    
def load_parser(task_type: str, valid_items: Optional[List[Any]], list_length: Optional[int]) -> Union[BaseModel, None]:
    if task_type == "Example Generation":
        class ExampleGenerationOutput(BaseModel):
            reasoning: str = Field(description="The thought process leading to the answer")
            
        return ExampleGenerationOutput
    
    elif task_type == "Classification":
        if list_length:
            class OutputParser(BaseModel):
                reasoning: str = Field(description=f"The thought process leading to the final answer.")
                label: List[Union[bool, int, float, str]] = Field(description=f"The final answer. This should be a list consistent of the options {valid_items}. The final list should have exactly {list_length} items.")
                
                @field_validator('label')
                @classmethod
                def validate_label(cls, value):
                    if not all(item in valid_items for item in value):
                        raise ValueError(f"The label must be from {valid_items}.")
                    if len(value) != list_length:
                        raise ValueError(f"The label must contain exactly {list_length} items.")
                    return value
        else:
            class OutputParser(BaseModel):
                reasoning: str = Field(description=f"The thought process leading to the final answer.")
                label: Union[bool, int, float, str] = Field(description=f"The final answer. The options are {valid_items}.")
                
                @field_validator('label')
                @classmethod
                def validate_label(cls, value):
                    if value not in valid_items:
                        raise ValueError(f"The label must be from {valid_items}.")
                    return value
        return OutputParser
    
    elif task_type == "Regression":
        if list_length:
            class OutputParser(BaseModel):
                reasoning: str = Field(description=f"The thought process leading to the final answer.")
                label: List[float] = Field(description=f"The final answer. This should be a list of floating point numbers. The final list should have exactly {list_length} items.")
                
                @field_validator('label')
                @classmethod
                def validate_label(cls, value):
                    if len(value) != list_length:
                        raise ValueError(f"The label must contain exactly {list_length} items.")
                    return value
        else:
            class OutputParser(BaseModel):
                reasoning: str = Field(description=f"The thought process leading to the final answer.")
                label: float = Field(description=f"The final answer. This should be a floating point number.")
            
        return OutputParser
    
    # TODO
    elif task_type == "Single Label Named Entity Recognition":
        class NamedEntityRecognitionOutput(BaseModel):
            reasoning: str = Field(description="The thought process leading to the answer")
            label: str = Field(description="The input text with the named entity highlighted in the format: [START] entity [END]")
        return NamedEntityRecognitionOutput
    elif task_type == "Multi Label Named Entity Recognition":
        return MultiLabelNamedEntityRecognitionOutput
    else:
        raise ValueError(f"Unknown task type: {task_type}")

