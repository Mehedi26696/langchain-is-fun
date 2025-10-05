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

text = "Bangladesh is a country in South Asia."

# Generate embeddings
vector = gemini_embeddings.embed_query(text)

print("Embedding length:", len(vector))
print(vector)
