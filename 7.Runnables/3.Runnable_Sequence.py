from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key=api_key
)

prompt1 = PromptTemplate(
    template="Tell me jokes about {topic}.",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Explain the jokes {jokes} in a simple way for a child to understand.",
    input_variables=["jokes"],
)

parser = StrOutputParser()

# chain = RunnableSequence([prompt, model, parser])  // Not works now
# Use the '|' operator to create a RunnableSequence

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "cats"})

print(result)