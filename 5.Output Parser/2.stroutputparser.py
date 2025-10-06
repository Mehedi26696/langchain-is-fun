from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template= 'Write a detailed report on the following topic. /n {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Summarize the following text in a concise manner: /n {text}',
    input_variables=['text']
)

 
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser


result = chain.invoke({'topic':'The impact of climate change on global agriculture.'})

print(result)