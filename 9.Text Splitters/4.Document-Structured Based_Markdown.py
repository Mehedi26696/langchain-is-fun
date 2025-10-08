from langchain.text_splitter import RecursiveCharacterTextSplitter,Language

text = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git

"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])


# chunk_size means the maximum number of characters in each chunk.
# chunk_overlap means the number of characters that overlap between chunks.
# The RecursiveCharacterTextSplitter tries to split the text using a hierarchy of separators.
# Here there is a markdown snippet, so it will try to split by headings, then by subheadings, then by paragraphs, then by lines, and finally by characters.
# This means that it will try to keep sections and paragraphs intact as much as possible.
# It will try to merge smaller chunks into larger ones recursively without exceeding the chunk_size.
# This helps to maintain the context and meaning of the text in each chunk.

