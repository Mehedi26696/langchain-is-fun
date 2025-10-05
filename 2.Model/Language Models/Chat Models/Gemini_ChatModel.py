from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
 
api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key  
)
 
result = model.invoke("What is the capital of Bangladesh?")

print(result.content)

# Gemini uses Google Cloud SDK internally.
# By default it looks for Application Default Credentials (ADC).
# If ADC is not set, We MUST explicitly provide the API key.
# Env variable: We can set any variable name in  .env file, just make sure to use the same name in os.getenv("VARIABLE_NAME")

