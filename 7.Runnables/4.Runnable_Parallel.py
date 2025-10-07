from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel
import os
load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key=api_key
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet':  prompt1 | model | parser,
    'linkedin': prompt2 | model | parser
})

result = parallel_chain.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linkedin'])