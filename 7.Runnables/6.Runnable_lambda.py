from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough , RunnableLambda
import os
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key=api_key
)

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

 

parser = StrOutputParser()


def word_count(text: str) -> int:
    return len(text.split())




jokes_chain = prompt1 | model | parser


parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})


final_chain = jokes_chain | parallel_chain

result = final_chain.invoke({'topic':'AI'})

print(result['joke'])
print(result['word_count'])