from  langchain_groq import ChatGroq

from dotenv import load_dotenv
from typing import  Annotated , Optional, Literal
from pydantic import BaseModel, Field, ValidationError, conint, constr , EmailStr
import os
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")


class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes mentioned in the review")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="The sentiment of the review (positive, negative, neutral)")
    pros: Optional[list[str]] = Field(None, description="List all the positive aspects mentioned in the review")
    cons: Optional[list[str]] = Field(None, description="List all the negative aspects mentioned in the review")
    name: Optional[str] = Field(None, description="Name of the reviewer if mentioned")


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Mehedi Hasan
""")

print(result)

print("Summary:", result.summary)
print("Sentiment:", result.sentiment)
print("Key Themes:", result.key_themes)
print("Pros:", result.pros)
print("Cons:", result.cons)
print("Reviewer Name:", result.name)