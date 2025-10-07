from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")


url = "https://www.startech.com.bd/apple-macbook-air-m2-13-inch"

loader = WebBaseLoader(url)

docs = loader.load()

# print(docs)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

chain = prompt | model | parser

result = chain.invoke({"question":"What is the price of the product?",
"text": docs[0].page_content})


print(result)
