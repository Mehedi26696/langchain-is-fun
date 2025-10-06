from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key  
    )

prompt1 = PromptTemplate(
    template = "Generate details report on the topic: {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Give 3 key takeaways from the following report: {report}.",
    input_variables=["report"]
)


parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Artificial Intelligence"})

print(result)


# Visualize the chain
chain.get_graph().print_ascii()



# Generate a detailed report on a topic.
# Extract key takeaways from that report.

# Chain Structure:
# prompt1 | model | parser | prompt2 | model | parser

# Step 1: Prompt 1 receives {"topic": "Artificial Intelligence"}.
# Step 2: Model Call 1 generates a detailed report on Artificial Intelligence.
# Step 3: Parser 1 extracts clean text from the model's output.
# Step 4: Prompt 2 takes that report as input and asks for 3 key takeaways.
#          Since Prompt 2 has a single input variable ("report"),
#          LangChain automatically maps the previous string output
#          to {"report": <string>}.
# Step 5: Model Call 2 generates the 3 key takeaways.
# Step 6: Parser 2 ensures a clean text result.
# Final Output: The chain returns the summarized key takeaways.

