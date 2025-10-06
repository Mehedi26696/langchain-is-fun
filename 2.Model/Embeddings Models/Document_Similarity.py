from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Football, also known as soccer in some countries, is the world's most popular sport.",
    "It is played by two teams of eleven players with a spherical ball.",
    "The objective is to score by getting the ball into the opposing goal.",
    "Major football events include the FIFA World Cup and UEFA Champions League."
]

query = "What is the main objective of football?"

query_vector = embedding.embed_query(query)

document_vectors = embedding.embed_documents(documents)

similarities = cosine_similarity([query_vector], document_vectors)

print(str(similarities))

scores = similarities[0]

print(list(enumerate(scores)))

ranked_docs = sorted(enumerate(documents), key=lambda x: scores[x[0]], reverse=True)
for idx, doc in ranked_docs:
    print(f"Score: {scores[idx]:.4f} - Document: {doc}")

