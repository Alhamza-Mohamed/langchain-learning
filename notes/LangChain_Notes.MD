<style>
/* General note styling */  
  body {
    font-size: 130%;
  }

    img {
    display: block;
    margin: 20px auto;
    max-width: 70%;
    }

    .math {
    text-align: center;
}
</style>

# LangChain  

## Table of content

- [LangChain](#langchain)
  - [Table of content](#table-of-content)
  - [Lesson 1: Models \& chat models](#lesson-1-models--chat-models)
    - [First LangChain chat model](#first-langchain-chat-model)
    - [Messages](#messages)

## Lesson 1: Models & chat models

The goal of this lesson is to learn what LangChain is, why it exists, and how modern chat models work using LangChain and Gemini.  

**What is LangChain?**  

LangChain is an orchestration framework for building LLM-powered applications, it does not replace the LLM.  

Instead, it provides abstraction for interacting with models, prompts, tools, retrieval systems, and workflows.  

**The problem LangChain solves**  

without LangChain:

```text
Application
      │
      ▼
Provider SDK
      │
      ▼
Gemini/OpenAI/Claude
```

Applications, provider SDK and the LLMs are separated, each provider has different APIs.  

Switching providers requires code changes.  

With LangChain:

```text
Application
      │
      ▼
LangChain
      │
      ▼
Gemini / OpenAI / Claude
```

The application interacts with a common interface.  

**When to use LangChain?**  
when building:

- AI Assistants.
- RAG.
- Agents.
- Tool Calling.
- Multi-step workflows.
- Production AI systems.

**LangChain should not be used to build:**

- Simple scripts.
- one API call.
- Tiny prototypes.

Sometimes the provider SKD alone is simpler.  

**Models vs chat models:**  

Old LLMs was receiving and sending texts

```text
Text
 ↓
Model
 ↓
Text
```

While modern models expects messages and sending the response also as messages

```text
Messages
    ↓
Chat Model
    ↓
Messages
```

Modern AI systems are conversation-based.  

### First LangChain chat model

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0,
    api_key=API_KEY,
)
```

**Ownership:**

LangChain

- ChatGoogleGenerativeAI
- invoke()

Gemini

- model
- inference
- temperature

Python

- Variables
- Imports

### Messages  

messages is an official LangChain term and it has common message types:

```text
HumanMessage

SystemMessage

AIMessage

ToolMessage
```

**SystemMessage**  
Defines the assistant's behavior

```python
SystemMessage(
    content="You are a senior Python instructor."
)
```

Should appear first in the conversation.

**HumanMessage**  
Represents a message sent by the user.

```python
HumanMessage(
    content="Explain LangChain."
)
```

**AIMessage**  
Returned by the model

```python
response = llm.invoke(messages)
```

`response` is not a string, it is `AIMessage`.  
The generated text is: `response.content`.  

**Mental model**  

Don't think about that the LLM applications is just LLM expects string and respond with string:

```text
String
  ↓
 LLM
  ↓
String
```

Instead think of it as a chat model that expects conversation and respond with conversation:

```text
Conversation
     ↓
Chat Model
     ↓
Conversation
```

This is how modern LLM applications work.

**Engineering takeaways**  

- LangChain is an orchestration framework.
- Chat models operate on messages.
- Messages carry roles, not just text.
- Keep configuration in `.env`.
- Separate Python, LangChain, and provider responsibilities.
- Build applications around conversations, not strings.  
