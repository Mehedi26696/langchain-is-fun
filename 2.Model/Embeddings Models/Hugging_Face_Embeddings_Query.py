from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Initialize HuggingFace Embeddings using Inference API
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"use_auth_token": hf_api_token}
)

text = "Bangladesh is a country in South Asia."

vector = hf_embeddings.embed_query(text)
print("Embedding length:", len(vector))
print(vector)




# If you use use_api=True, then the Hugging Face API token is required. This is because your code will use the Hugging Face Inference API, which needs authentication.

# If you set use_api=False, the model will run locally and you do not need the token (but you must have the model downloaded or it will be downloaded automatically).

# So, the token is only required if use_api=True.
