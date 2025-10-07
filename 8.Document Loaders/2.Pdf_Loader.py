from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


loader = PyPDFLoader("Document Loaders.pdf")

docs = loader.load()

print(docs)
print(type(docs))
print(len(docs))
print(docs[0])
print(type(docs[0]))
print(docs[0].page_content)
print(docs[0].metadata)


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=api_key 
)


prompt = PromptTemplate(
    template="Summarize the following text: {text}",
    input_variables=["text"]
)

parser = StrOutputParser()


chain = prompt | model | parser

result = chain.invoke({"text": docs[0].page_content})

print(result)
