from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

model1 = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key  
    )

model2 = ChatGroq(model="llama-3.1-8b-instant")

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)
parser = StrOutputParser()

parallel_chain =  RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain
text = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. It encompasses a wide range of technologies, including machine learning, natural language processing, computer vision, and robotics. AI has the potential to revolutionize various industries by automating tasks, improving decision-making processes, and enhancing customer experiences.

Over the past decade, advancements in AI have accelerated rapidly, leading to the development of systems that can perform complex tasks such as language translation, image recognition, and even autonomous driving. Machine learning, a subset of AI, enables computers to learn from data and improve their performance over time without being explicitly programmed. Deep learning, which uses neural networks with many layers, has been particularly successful in areas like speech recognition and image analysis.

Despite its many benefits, AI also raises important ethical and societal questions. Issues such as job displacement, data privacy, algorithmic bias, and the potential misuse of AI technologies are topics of ongoing debate. As AI continues to evolve, it is crucial for researchers, policymakers, and industry leaders to work together to ensure that AI is developed and used responsibly for the benefit of society.
"""

result = chain.invoke({"text": text})

print(result)


# Visualize the chain
chain.get_graph().print_ascii()





# This pipeline demonstrates a parallel processing workflow in LangChain using two LLMs (Gemini and LLaMA),
# where different models handle separate tasks simultaneously and their outputs are merged in the end.

# Chain Structure:
# RunnableParallel({"notes": prompt1 | model1 | parser,
#                   "quiz":  prompt2 | model2 | parser}) 
# → prompt3 | model1 | parser

# Step 1: Load environment variables (for Gemini API key) using dotenv.
# Step 2: Initialize two chat models:
#          - model1: Google Gemini 2.5 Flash (used for generating notes and merging results)
#          - model2: Groq LLaMA 3.1 8B Instant (used for generating quiz questions)
# Step 3: Define three PromptTemplates:
#          - prompt1: "Generate short and simple notes" → input variable: {text}
#          - prompt2: "Generate 5 short question answers" → input variable: {text}
#          - prompt3: "Merge notes and quiz into one document" → input variables: {notes, quiz}
# Step 4: Define a StrOutputParser to extract clean text from each model output.

# Step 5: Create a parallel chain using RunnableParallel:
#          - The same input text is fed to both prompt1 and prompt2 in parallel.
#          - model1 handles note generation; model2 handles quiz generation.
#          - The parser cleans both outputs.
#          The resulting dictionary looks like:
#          {
#              "notes": "<generated notes text>",
#              "quiz": "<generated quiz text>"
#          }

# Step 6: Define a merge chain (prompt3 | model1 | parser):
#          - prompt3 takes the two outputs ("notes" and "quiz") as input.
#          - model1 (Gemini) merges them into a single formatted document.
#          - parser ensures clean final text output.

# Step 7: Combine both parts:
#          parallel_chain | merge_chain
#          This means:
#          1. Run note and quiz generation in parallel.
#          2. Pass both results into the merge prompt to produce a unified document.

# Step 8: Provide input text (about Artificial Intelligence).
#          The text is fed into the parallel chain.
#          The parallel step returns both "notes" and "quiz" contents,
#          which are then merged by the final Gemini call.

# Step 9: Invoke the full chain and print the final merged result.
#          The result contains concise AI notes and 5 short Q&A pairs
#          combined into a single well-structured document.

# Step 10: Visualize the pipeline using ASCII graph to understand the chain structure:
#           chain.get_graph().print_ascii()
#           This shows how each prompt, model, and parser is connected.

# Final Output: A unified document combining AI notes and quiz questions,
#                  generated collaboratively by two different models running in parallel.
