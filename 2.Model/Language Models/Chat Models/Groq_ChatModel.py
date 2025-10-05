from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")
 
result = model.invoke("What is the capital of Bangladesh?")

print(result.content)


# Groq automatically reads the API key from environment
# We do NOT need to pass it manually
# Env variable: GROQ_API_KEY
