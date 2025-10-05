from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")  # Must pass API key

# Initialize Claude model
model = ChatAnthropic(
    model="claude-4-sonnet",
    temperature=0.7,
    api_key=api_key
)


response = model.invoke("What is the capital of Bangladesh?")
print(response.content)
