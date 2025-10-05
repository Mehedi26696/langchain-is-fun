from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
 
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.0,
    max_tokens=512,
)

result = llm.invoke("What is the capital of Bangladesh?")
print(result.content)
