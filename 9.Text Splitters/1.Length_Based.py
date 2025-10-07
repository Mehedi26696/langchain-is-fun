from langchain.text_splitter import CharacterTextSplitter

text = """
Artificial Intelligence (AI) is transforming the way we interact with technology. From virtual assistants to recommendation systems, AI is becoming an integral part of our daily lives. Its ability to process vast amounts of data enables smarter and faster decision-making.

In recent years, advancements in machine learning and deep learning have accelerated AI development. These technologies allow computers to learn from experience and improve over time. As a result, AI systems are now capable of performing complex tasks such as image recognition and natural language processing.

Despite its many benefits, AI also presents challenges and ethical considerations. Issues like data privacy, algorithmic bias, and job displacement need to be addressed. Ongoing research and collaboration are essential to ensure AI is developed responsibly and benefits society as a whole.
"""

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=""
)

result = splitter.split_text(text)

print(result)


#  chunk_size means the maximum number of characters in each chunk.
#  chunk_overlap means the number of characters that overlap between chunks.
#  separator means the character used to split the text. If empty, it will split by character.
#  If the text is shorter than chunk_size, it will return the whole text as a single chunk.