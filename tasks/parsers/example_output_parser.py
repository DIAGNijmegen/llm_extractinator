from pydantic import BaseModel


class Product(BaseModel):
    name: str
    price: float


class OutputParser(BaseModel):
    products: list[Product]
