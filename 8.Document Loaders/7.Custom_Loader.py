from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document

class CustomTextLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # You can add custom parsing logic here
        return [Document(page_content=text, metadata={"source": self.file_path})]

# Example usage:
if __name__ == "__main__":
    
    loader = CustomTextLoader("Football.txt")
    docs = loader.load()
    for doc in docs:
        print(doc.page_content)