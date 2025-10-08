from langchain_experimental.text_splitter import SemanticChunker

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_api_key
)


semantic_splitter = SemanticChunker(
    embeddings=gemini_embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.0,
)

text = """Education: Education provides a foundation of knowledge and critical thinking; lessons in teamwork, discipline, and strategic thinking learned in the classroom translate to performance on the pitch. Students learn to analyze plays like problems, to communicate clearly under pressure, and to value practice and feedback—skills that serve both academic achievement and personal growth.Football: Coaching acts as mentorship, where tactical instruction, drills, and collaborative goal-setting reinforce perseverance and resilience, linking athletic development with lifelong learning.

A river is a living thread across the landscape, carving valleys and carrying nutrients that sustain diverse ecosystems. Its surface reflects the changing sky while its currents shape the earth beneath, supporting plants, animals, and human communities along its banks. Rivers connect regions, provide water for agriculture and industry, and offer tranquil spaces for recreation and reflection, reminding us of nature's continual flow and renewal."""

chunks =  semantic_splitter.split_text(text)

print(f"Number of chunks: {len(chunks)}")

print(chunks[0])


# Here semantic splitting has been done based on the meaning of the text.
# For that we have used the Gemini Embeddings model from Google Generative AI.
# The text has been split into chunks based on the semantic meaning of the text.
# breakpoint_threshold_type can be set to "standard_deviation" or "percentile" and it determines how the chunks are created.
# breakpoint_threshold_amount can be set to any float value. Higher the value, lesser the chunks


