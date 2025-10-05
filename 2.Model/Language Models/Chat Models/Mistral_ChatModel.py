from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os

 
load_dotenv()
 
api_key = os.getenv("MISTRAL_API_KEY")

# Initialize the Mistral model
model = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.7,
    api_key=api_key  # Must pass API key even for free tier
)

result = model.invoke("What is the capital of Bangladesh?")

 
print(result.content)

# Notes:
# - Mistral requires an API key for authentication, even on the free tier.
# - Env variable: We can name it anything in the .env file, just make sure to use the same name in os.getenv("VARIABLE_NAME").
# - This is similar to how Gemini requires a Google API key to work when ADC is not set.
