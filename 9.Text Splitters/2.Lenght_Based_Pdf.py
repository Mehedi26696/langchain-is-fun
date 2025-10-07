from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Text Splitters.pdf")

documents = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=""
)

result = splitter.split_documents(documents)

# print(result)

print(result[0].page_content)

#  chunk_size means the maximum number of characters in each chunk.
#  chunk_overlap means the number of characters that overlap between chunks.
#  separator means the character used to split the text. If empty, it will split by character.
#  If the text is shorter than chunk_size, it will return the whole text as a single chunk.