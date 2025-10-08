from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


text = """
class Student:
    def __init__(self, name, age, grades):
        self.name = name
        self.age = age
        self.grades = grades

    def is_passing(self):
        return sum(self.grades) / len(self.grades) >= 60

    def __str__(self):
        status = "Passing" if self.is_passing() else "Failing"
        return f"Student: {self.name}, Age: {self.age}, Status: {status}"
    def add_grade(self, grade):
        self.grades.append(grade)
    def average_grade(self):
        return sum(self.grades) / len(self.grades) if self.grades else 0
"""

# Create a splitter configured for English text
splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=120,
    chunk_overlap=20,
    language=Language.PYTHON,
)
 
# Split the text into chunks
chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])


# chunk_size means the maximum number of characters in each chunk.
# chunk_overlap means the number of characters that overlap between chunks.
# The RecursiveCharacterTextSplitter tries to split the text using a hierarchy of separators.
# Here there is a python snippet, so it will try to split by classes, then by functions, then by lines, and finally by characters.
# This means that it will try to keep functions and classes intact as much as possible.
# It will try to merge smaller chunks into larger ones recursively without exceeding the chunk_size.