from pydantic import BaseModel, Field, ValidationError, conint, constr , EmailStr
from typing import Optional, Annotated

class ProductReview(BaseModel):
    # Basic fields with types and constraints
    product_name: str

    # Default values 
    rating: int = 5  # Default rating if not provided

    # Optional fields
    reviewer: Optional[str] = None
    price: float
     
    # Built-in validators

    email: EmailStr


    # Feilds with annotations and constraints
    
    quantity: Annotated[int, Field(gt=0, lt=1000, description="Quantity must be between 1 and 999")]
    sentiment: Annotated[str, constr(pattern="^(positive|negative|neutral)$")]
    summary: str = Field(
        default="No summary provided.",
        min_length=10,
        max_length=200,
        description="A short summary of the review (10–200 chars)."
    )

if __name__ == "__main__":
    try:
        data = {
            "product_name": "AI Headphones",
            "rating": "4",  # string, will be coerced to int
            "price": "99.9",  # string, coerced to float
            "quantity": 2,
            "sentiment": "positive",
            "summary": "Great sound quality and comfort!"
        }

        review = ProductReview(**data)

        print("DICT OUTPUT:")
        print(review.model_dump(), "\n")

        print("JSON OUTPUT:")
        print(review.model_dump_json(indent=4), "\n")

    except ValidationError as e:
        print("VALIDATION ERROR:")
        print(e.json())

