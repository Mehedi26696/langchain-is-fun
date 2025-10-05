from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Load Google API key from environment variables
google_api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini Embeddings
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_api_key
)

documents = [
    "Football, also known as soccer in some countries, is the world's most popular sport.",
    "It is played by two teams of eleven players with a spherical ball.",
    "The objective is to score by getting the ball into the opposing goal.",
    "Major football events include the FIFA World Cup and UEFA Champions League."
]

# Generate embeddings
vector = gemini_embeddings.embed_documents(documents)

print("Embedding length:", len(vector))
print(vector)
