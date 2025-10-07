from langchain_community.document_loaders import UnstructuredPDFLoader

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

loader = UnstructuredPDFLoader("Scanned.pdf")
docs = loader.load()
print(docs)
print(type(docs))
print(len(docs))
print(docs[0])
print(type(docs[0]))
print(docs[0].page_content)
print(docs[0].metadata)