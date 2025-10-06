from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


parser = JsonOutputParser()


template = PromptTemplate(
    template= 'Give me a product information including name, price, and description for the product: {product_name}. \n {format_instructions}',
    input_variables=['product_name'],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


prompt = template.invoke({'product_name':'laptop'})

result = model.invoke(prompt)

parsed_output = parser.parse(result.content)

print(parsed_output)
