from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Football, also known as soccer in some countries, is the world's most popular sport.",
    "It is played by two teams of eleven players with a spherical ball.",
    "The objective is to score by getting the ball into the opposing goal.",
    "Major football events include the FIFA World Cup and UEFA Champions League."
]

result = embedding.embed_documents(documents)

print(str(result))