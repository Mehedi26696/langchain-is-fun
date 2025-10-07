from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Artificial Intelligence (AI) is transforming the way we interact with technology. From virtual assistants to recommendation systems, AI is becoming an integral part of our daily lives. Its ability to process vast amounts of data enables smarter and faster decision-making.

In recent years, advancements in machine learning and deep learning have accelerated AI development. These technologies allow computers to learn from experience and improve over time. As a result, AI systems are now capable of performing complex tasks such as image recognition and natural language processing.

Despite its many benefits, AI also presents challenges and ethical considerations. Issues like data privacy, algorithmic bias, and job displacement need to be addressed. Ongoing research and collaboration are essential to ensure AI is developed responsibly and benefits society as a whole.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0 
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)


#  chunk_size means the maximum number of characters in each chunk.
#  chunk_overlap means the number of characters that overlap between chunks.
#  The RecursiveCharacterTextSplitter tries to split the text using a hierarchy of separators.
#  It first tries to split by double newlines, then by single newlines, then by spaces, and finally by characters.
#  This means that it will try to keep sentences and paragraphs intact as much as possible.
#  It will try to merge smaller chunks into larger ones recursively without exceeding the chunk_size.
#  This helps to maintain the context and meaning of the text in each chunk.
#  If the text is shorter than chunk_size, it will return the whole text as a single chunk.
#  This method is more sophisticated than the CharacterTextSplitter, which simply splits by characters.
#  It is especially useful for longer texts where maintaining context is important.
