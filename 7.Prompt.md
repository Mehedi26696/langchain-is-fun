# LangChain Prompting Guide

A quick reference to effective prompt patterns in LangChain.

---

## Overview

- **Dynamic prompts:** Parameterize templates for flexibility and reuse.
- **Role-based prompts:** Structure conversations for chat-based LLMs.
- **Few-shot prompts:** Provide examples to clarify expected outputs.

**Inputs:** Templates, variables, examples  
**Outputs:** Formatted strings/messages for LLMs  
**Common errors:** Missing variables, prompt too long, incorrect role order  
**Goal:** Prompts are well-formatted and pass tests

---

## 1. Prompt Basics: `PromptTemplate`

A `PromptTemplate` is the core building block for reusable prompts. Define templates with named variables and fill them at runtime.

**Example:**

```python
from langchain.prompts import PromptTemplate

template = """
Summarize the following text in {style} (max {max_tokens} words):

{text}
"""

prompt = PromptTemplate(
    input_variables=["text", "style", "max_tokens"],
    template=template,
)

filled = prompt.format(text="LangChain is...", style="concise", max_tokens=50)
print(filled)
```

**Tips:**
- Keep templates focused and concise.
- Validate required `input_variables` with unit tests.
- Use `.partial()` to pre-fill some variables for reuse:

```python
partial_prompt = prompt.partial(style="concise")
# Only text and max_tokens are now required
filled = partial_prompt.format(text="...", max_tokens=50)
```

---

## 2. Dynamic & Reusable Prompts

Dynamic prompts allow you to reuse templates by passing variables at runtime.

- Parameterize style, tone, or length for flexibility.
- Compose small templates for instructions and content.
- Store and load templates by name for easy reuse.

**Example:**

```python
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Summarize {topic} in a {emotion} tone")
print(prompt.format(topic="Cricket", emotion="fun"))
```

---

## 3. Role-Based (Chat) Prompts

Use `ChatPromptTemplate` to structure prompts as message sequences for chat LLMs.

**Example:**

```python
from langchain.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {profession}."),
    ("user", "Tell me about {topic}."),
])

msgs = chat_prompt.format_messages(profession="Doctor", topic="Viral Fever")
for msg in msgs:
    print(f"{msg.type}: {msg.content}")
```

**Tips:**
- Place instructions in the `system` message.
- Keep user messages clear and concise.
- Reuse system templates for consistency.

---

## 4. Few-Shot Prompting

Few-shot prompting uses examples to guide the LLM’s output. Use `FewShotPromptTemplate` to structure these examples.

**Example:**

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

example = PromptTemplate.from_template("Q: {question}\nA: {answer}\n")
examples = [
    {"question": "What is LangChain?", "answer": "A library for LLM apps."},
    {"question": "What's a prompt template?", "answer": "A reusable string with variables."},
]

fewshot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example,
    prefix="You are a helpful assistant.",
    suffix="Q: {input}\nA:",
    input_variables=["input"],
)

print(fewshot.format(input="How does LangChain help with prompts?"))
```

**Tips:**
- Use 2–8 high-quality, concise examples.
- Keep examples short to save tokens.

---

**Summary:**  
LangChain’s prompt templates help you build flexible, reusable, and testable prompts for LLMs. Use dynamic variables, role-based structures, and few-shot examples to improve clarity and output quality.
