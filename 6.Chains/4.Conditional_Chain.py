from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch , RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser

import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model1 = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key  
    )

model2 = ChatGroq(model="llama-3.1-8b-instant")


class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser1 = PydanticOutputParser(pydantic_object=Feedback)

parser2 = StrOutputParser()


prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser1.get_format_instructions()}
)

classifier_chain = prompt1 | model1 | parser1


prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model2 | parser2),
    (lambda x:x.sentiment == 'negative', prompt3 | model2 | parser2),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "The product quality is excellent and I am very satisfied with my purchase."})

print(result)

result = chain.invoke({"feedback": "The product quality is poor and I am very disappointed with my purchase."})
print(result)


# Visualize the chain
chain.get_graph().print_ascii()


# This pipeline demonstrates conditional routing in LangChain using RunnableBranch.
# It classifies the sentiment of a feedback message (positive or negative)
# and dynamically chooses an appropriate response-generation path based on the result.

# Chain Structure:
# (prompt1 | model1 | parser1) → RunnableBranch([...])
# → (prompt2 | model2 | parser2) or (prompt3 | model2 | parser2)

# Step 1: Load environment variables (GEMINI_API_KEY) using dotenv for secure key management.

# Step 2: Initialize two chat models:
#          - model1: Google Gemini 2.5 Flash — used for sentiment classification.
#          - model2: Groq LLaMA 3.1 8B Instant — used for generating feedback responses.

# Step 3: Define a Pydantic model "Feedback" with a single field:
#          sentiment ∈ {'positive', 'negative'}
#          This schema enforces structured output for sentiment classification.

# Step 4: Create two parsers:
#          - parser1: PydanticOutputParser → validates and parses model1 output into a Feedback object.
#          - parser2: StrOutputParser → converts model2 output into plain text.

# Step 5: Define the first prompt (prompt1) for sentiment classification:
#          Template: "Classify the sentiment of the following feedback..."
#          Input variable: {feedback}
#          Partial variable: {format_instruction} from parser1 ensures the model returns valid JSON
#          matching the Pydantic schema.

# Step 6: Build the sentiment classifier chain:
#          prompt1 | model1 | parser1
#          → Takes feedback text as input.
#          → Gemini classifies sentiment as 'positive' or 'negative'.
#          → Output is parsed into a validated Feedback object.

# Step 7: Define two response-generation prompts:
#          - prompt2: Used when sentiment is positive → “Write an appropriate response to this positive feedback.”
#          - prompt3: Used when sentiment is negative → “Write an appropriate response to this negative feedback.”

# Step 8: Create a conditional branch using RunnableBranch:
#          RunnableBranch dynamically routes the flow based on the classifier output.
#          - If sentiment == 'positive' → executes prompt2 | model2 | parser2
#          - If sentiment == 'negative' → executes prompt3 | model2 | parser2
#          - Otherwise → RunnableLambda returns "could not find sentiment"

# Step 9: Combine both parts:
#          classifier_chain | branch_chain
#          This means:
#          1. The first chain classifies the sentiment using Gemini.
#          2. The second chain (branch) chooses the appropriate LLaMA prompt to generate a suitable reply.

# Step 10: Invoke the chain with a positive feedback example:
#          Input: "The product quality is excellent and I am very satisfied with my purchase."
#          - Classifier identifies sentiment as "positive".
#          - RunnableBranch selects prompt2 → model2 → parser2.
#          - Output: A polite response to positive feedback.

# Step 11: Invoke the chain again with a negative feedback example:
#          Input: "The product quality is poor and I am very disappointed with my purchase."
#          - Classifier identifies sentiment as "negative".
#          - RunnableBranch selects prompt3 → model2 → parser2.
#          - Output: A courteous apology or resolution message.

# Step 12: Visualize the entire chain structure using ASCII graph:
#          chain.get_graph().print_ascii()
#          This displays how the classifier connects to conditional branches for response generation.

# Final Output:
# For positive feedback → A friendly appreciation message.
# For negative feedback → A professional and empathetic apology response.
# The chain showcases the power of conditional logic and structured parsing in LangChain workflows.

