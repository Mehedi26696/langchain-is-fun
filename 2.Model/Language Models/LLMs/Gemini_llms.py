from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini LLM
model = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key
)

# Invoke the LLM with a prompt
result = model.invoke("Write a short poem about Bangladesh in 2 lines.")

# Print the result
print(result)

# Notes:
# - This uses plain text prompt instead of chat messages
# - google_api_key must be passed if ADC is not set
# - Works like a standard LLM (no roles needed)
