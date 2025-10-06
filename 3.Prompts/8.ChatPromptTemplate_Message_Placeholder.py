from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

google_api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=google_api_key
    )


# chat template

chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful customer support assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",  "{query}"),
    ]
)



# load chat history
chat_history = []
with open("chat_history.txt", "r") as file:
    chat_history.extend(file.readlines())

print(chat_history)


# create prompt
prompt = chat_template.invoke(
    {
        "chat_history": chat_history,
        "query": "Where is my refund?"
    }
)
print(prompt)


# get response
result = model.invoke(prompt)
print(result.content)

