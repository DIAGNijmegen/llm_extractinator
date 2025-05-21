from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class OutputParser(BaseModel):
    string_field: str
    optional_string_field: Optional[str] = None

    int_field: int
    optional_int_field: Optional[int] = None

    float_field: float
    optional_float_field: Optional[float] = None

    bool_field: bool
    optional_bool_field: Optional[bool] = None

    any_field: Any
    optional_any_field: Optional[Any] = None

    list_field: List[str]
    optional_list_field: Optional[List[int]] = None

    dict_field: Dict[str, Any]
    optional_dict_field: Optional[Dict[str, str]] = None

    literal_string_field: Literal["A", "B", "C"]
    optional_literal_string_field: Optional[Literal["X", "Y", "Z"]] = None

    literal_int_field: Literal[1, 2, 3]
    optional_literal_int_field: Optional[Literal[4, 5, 6]] = None
