from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Football, also known as soccer in some countries, is the world's most popular sport.",
    "It is played by two teams of eleven players with a spherical ball.",
    "The objective is to score by getting the ball into the opposing goal.",
    "Major football events include the FIFA World Cup and UEFA Champions League."
]

vector = embedding.embed_documents(documents)

print(str(vector))