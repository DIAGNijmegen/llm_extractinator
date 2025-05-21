from typing import Literal

from pydantic import BaseModel


class OutputParser(BaseModel):
    HR: int
    Name: Literal["Alice", "Bob", "Charlie", "David", "Emma"]
