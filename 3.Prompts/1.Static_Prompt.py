from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

import os
import streamlit as st

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key  
    )

st.header("Research Assistant using Gemini-2.5-Flash")

user_input = st.text_input("Enter your prompt:")

if st.button("summarize")  and user_input:
    result = model.invoke(user_input)
    st.write(result.content)
    
     