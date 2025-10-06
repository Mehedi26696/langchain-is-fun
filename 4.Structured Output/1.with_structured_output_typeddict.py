from  langchain_groq import ChatGroq

from dotenv import load_dotenv
from typing import TypedDict
import os
load_dotenv()

model = ChatGroq(model="openai/gpt-oss-20b")

class Review(TypedDict):
    summary: str
    sentiment: str



structured_model = model.with_structured_output(Review)

result= structured_model.invoke(
     """I recently purchased this product and I am extremely satisfied with its performance. The build quality is excellent, and it exceeded my expectations in every way. I would highly recommend it to anyone looking for a reliable and efficient solution."""
)

print(result)

print("Summary:", result['summary'])
print("Sentiment:", result['sentiment'])

# Gemini 2.5 flash does not currently support with_structured_output() when we pass a TypedDict directly.
# We have to create a new model instance using the with_structured_output() method.