from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

 
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


result = model.invoke("Tell me a joke.")

print(result.content)
