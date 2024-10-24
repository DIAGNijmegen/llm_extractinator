from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, Type
from pydantic import BaseModel, Field, create_model
from typing import Literal, get_args

type_mapping = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "any": Any
}

def create_field(field_info: Dict[str, Any]) -> Any:
    """
    Create a Pydantic field with a type, description, and optional Literal values.
    """
    # Determine if the field is optional
    is_optional = field_info.get("optional", False)
    description = field_info.get("description", None)
    
    # Handle nested object (dictionary) types
    if field_info["type"] == "dict":
        nested_model = create_pydantic_model_from_json(
            field_info["properties"],
            model_name="DictionaryModel"
        )
        field_type = nested_model
    else:
        # Get the basic type from the type mapping
        field_type = type_mapping.get(field_info["type"], Any)

        # Handle literals if specified
        literals = field_info.get("literals")
        if literals:
            field_type = Literal[tuple(literals)]
    
    # If the field is optional, wrap the type with Optional
    if is_optional:
        field_type = Optional[field_type]
    
    # Create a Pydantic Field with a description
    return (field_type, Field(default=None if is_optional else ..., description=description))


def create_pydantic_model_from_json(data: Dict[str, Any], model_name: str = 'OutputParser') -> Type[BaseModel]:
    fields = {}
    for key, field_info in data.items():
        fields[key] = create_field(field_info)
    
    return create_model(model_name, **fields)
        
def load_parser(task_type: str, parser_format: Optional[Dict[str, Any]]) -> Type[BaseModel]:
    if task_type == "Example Generation":
        class ExampleGenerationOutput(BaseModel):
            reasoning: str = Field(description="The thought process leading to the answer")
        return ExampleGenerationOutput
    else:
        return create_pydantic_model_from_json(data=parser_format)
    
    
    
    # elif task_type == "Classification":
    #     if list_length:
    #         class OutputParser(BaseModel):
    #             reasoning: str = Field(description=f"The thought process leading to the final answer.")
    #             label: List[Union[bool, int, float, str]] = Field(description=f"The final answer. This should be a list consistent of the options {valid_items}. The final list should have exactly {list_length} items.")
                
    #             @field_validator('label')
    #             @classmethod
    #             def validate_label(cls, value):
    #                 if not all(item in valid_items for item in value):
    #                     raise ValueError(f"The label must be from {valid_items}.")
    #                 if len(value) != list_length:
    #                     raise ValueError(f"The label must contain exactly {list_length} items.")
    #                 return value
    #     else:
    #         class OutputParser(BaseModel):
    #             reasoning: str = Field(description=f"The thought process leading to the final answer.")
    #             label: Union[bool, int, float, str] = Field(description=f"The final answer. The options are {valid_items}.")
                
    #             @field_validator('label')
    #             @classmethod
    #             def validate_label(cls, value):
    #                 if value not in valid_items:
    #                     raise ValueError(f"The label must be from {valid_items}.")
    #                 return value
    #     return OutputParser
    
    # elif task_type == "Regression":
    #     if list_length:
    #         class OutputParser(BaseModel):
    #             reasoning: str = Field(description=f"The thought process leading to the final answer.")
    #             label: List[float] = Field(description=f"The final answer. This should be a list of floating point numbers. The final list should have exactly {list_length} items.")
                
    #             @field_validator('label')
    #             @classmethod
    #             def validate_label(cls, value):
    #                 if len(value) != list_length:
    #                     raise ValueError(f"The label must contain exactly {list_length} items.")
    #                 return value
    #     else:
    #         class OutputParser(BaseModel):
    #             reasoning: str = Field(description=f"The thought process leading to the final answer.")
    #             label: float = Field(description=f"The final answer. This should be a floating point number.")
            
    #     return OutputParser
    
    # # TODO
    # elif task_type == "Named Entity Recognition":
    #     class NamedEntityRecognitionOutput(BaseModel):
    #         reasoning: str = Field(description="The thought process leading to the answer")
    #         label: str = Field(description="The input text with the named entity highlighted in the format: [START] entity [END]")
    #     return NamedEntityRecognitionOutput
    
    # ### WRITE YOUR CUSTOM PARSER HERE ###
    # elif task_type == "Custom_Name":
    #     class OutputParser(BaseModel):
    #         variable1: str = Field(description="Description of variable1")
    #         variable2: int = Field(description="Description of variable2")
    #         variable3: List[str] = Field(description="Description of variable3")
            
    #         @field_validator('variable1')
    #         @classmethod
    #         def custom_validator(cls, value):
    #             if value not in ["option1", "option2"]:
    #                 raise ValueError("variable1 must be either 'option1' or 'option2'")
    #             return value
    #     return OutputParser
        
    # else:
    #     raise ValueError(f"Unknown task type: {task_type}")

