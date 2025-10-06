from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key
    )

chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful {domain} expert."),
        ("human", "Explain in simple terms, what is {topic}?"),
    ]
)

prompt = chat_template.invoke(
    {
        "domain": "machine learning",
        "topic": "transformers"
    }
)

print(prompt)

result = model.invoke(prompt)
print(result.content)
