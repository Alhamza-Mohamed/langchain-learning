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
  - [Lesson 2: Prompt templates](#lesson-2-prompt-templates)
    - [Why prompt templates exist?](#why-prompt-templates-exist)
    - [Prompt vs prompt template](#prompt-vs-prompt-template)
    - [Chat prompt template](#chat-prompt-template)
    - [MessagesPlaceholder](#messagesplaceholder)
  - [Lesson 3: Output parsers](#lesson-3-output-parsers)
    - [Why Output parsers exist?](#why-output-parsers-exist)
    - [Parsing Lifecycle](#parsing-lifecycle)
    - [StrOutputParser](#stroutputparser)
    - [JsonOutputParser](#jsonoutputparser)
  - [Lesson 4: Structured output with pydantic](#lesson-4-structured-output-with-pydantic)
    - [LangChain + pydantic flow](#langchain--pydantic-flow)
    - [BaseModel](#basemodel)
    - [JSON vs python dictionary](#json-vs-python-dictionary)
    - [Validation](#validation)
    - [Field](#field)
    - [PydanticOutputParser](#pydanticoutputparser)
    - [get\_format\_instructions()](#get_format_instructions)
    - [JsonOutputParser vs PydanticOutputParser](#jsonoutputparser-vs-pydanticoutputparser)
    - [Choosing the right parser](#choosing-the-right-parser)
    - [Key Takeaways](#key-takeaways)

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

## Lesson 2: Prompt templates

The goal of this lesson is to learn how LangChain separates prompt construction from model invocation by using reusable prompt templates instead of manually creating message.  

---

### Why prompt templates exist?

Without prompt templates:

```Python
message = [
  SystemMessage(
    content="You are a python instructor."
  ),
  HumanMessage(
    content="Explain decorators."
  ),
]
```

Every new request requires rebuilding the conversation manually.  

Prompt templates separate the conversation structure and variable values.  

---

### Prompt vs prompt template

Prompt: it is the final conversation sent to the LLM.  

Example:

```text
System:
You are a Python instructor.

Human:
Explain decorators.
```

Prompt template: it is a blueprint that contain placeholders.  

Example:

```text
System:
You are a Python instructor.

Human:
Explain {topic}.
```

The placeholders are replaced later when rendering the template.  

---

### Chat prompt template  

Create a reusable conversation template

```Python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a senior Python instructor."),
        ("human", "Explain {topic}."),
    ]
)

```

---

**Ownership**  

Python

- Variables.
- Dictionaries.
- Imports.

LangChain

- ChatPromptTemplate.
- Prompt rendering.
- Variable validation.

Gemini  
Nothing. Gemini only receives the final rendered conversation.  

---

**Rendering a prompt**  
A template is converted into a real conversation using:

```Python
prompt_value = prompt.invoke(
  {
    "topic": "decorators",
  }
)
```

This does NOT call the LLM. It only replace the placeholders.

**ChatPromptValue**  

Rendering returns `ChatPromptValue` Not" (String, list or HumanMessage).  
It is another LangChain object representing a completed prompt.  

The contained message can be inspected using `prompt_value.messages`.  

---

**Prompt pipeline**  

```text
ChatPromptTemplate
  ↓
invoke()
  ↓
ChatPromptValue
  ↓
LLM
  ↓
AIMessage
```

Prompt construction and model inference are separate steps.  

---

**Multiple variables**  

Templates can contain multiple placeholders.

```Python
("human", "Explain {topic} for a {level}.")
```

Rendering:

```Python
prompt.invoke(
    {
        "topic": "decorators",
        "level": "beginner",
    }
)
```

LangChain validates required variable before rendering (**Variable validation**).  

Missing variable:

```Python
{
    "topic": "decorators"
}
```

Raises an error because `{level}` is missing.  

Extra variables:

```Python
{
    "topic": "decorators",
    "level": "beginner",
    "language": "English",
}
```

Succeeds, unused variables are ignored.  

---

**Input_variables**  

Every templates stores the variables it expects.

```Python
print(prompt.input_variables)
```

Example: `['topic', 'level']`  

These variables are determined when the template is created, not every time `invoke()` is called.

---

**Why a Dictionary?**  

LangChain uses dictionary in rendering prompts instead of just direct strings.  
The benefits:

- Self-documenting
- Order-independent
- Easier validation
- Consistent with the rest of LangChain pipelines

---

### MessagesPlaceholder

The problem is A normal placeholder can replace only one string.  
It cannot replace an entire conversation.  

The solution is LangChain MessagesPlaceholder.  

```Python
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
```  

---

**Conversation history**  

History is a list of messages.

```Python
history = [
    HumanMessage(
        content="What is LangChain?"
    ),
    AIMessage(
        content="LangChain is a framework..."
    ),
]
```  

Rendering:

```Python
prompt.invoke(
    {
        "history": history,
        "question": "Can you explain it more simply?",
    }
)
```  

This produces the final prompt that sent to the model, which is a sequence of message objects:

```text
System
  ↓
Human
  ↓
 AI
  ↓
Human
```

---

**Why we don't use one big string?**  

Instead of using using:

```Python
history = """
Human...
AI...
Human...
"""
```

LangChain uses message objects because they preserve roles, conversation order and message structure.  
This is why lesson 1 focused on `HumanMessage`, `AIMessage`, and `SystemMessage`.  

## Lesson 3: Output parsers

Output parsers are LangChain runnables that transform the output of an LLM into a more useful python object.  
They also provide formatting instructions to guide the LLM toward producing outputs that can be successfully parsed.

### Why Output parsers exist?

LLMs naturally generate free-form text.  
Applications usually require structured data.

Instead of manually parsing responses in every application, LangChain provides reusable Output parsers.  

### Parsing Lifecycle

```text
Prompt
    │
    ▼
Output Parser
(get_format_instructions())
    │
    ▼
   LLM
    │
    ▼
AIMessage
    │
    ▼
Output Parser
(parse/invoke)
    │
    ▼
Python Object
```

The parser participates  both **before** and **after** the LLM.

---

### StrOutputParser

Its purpose is to return the model output as plain text.

- Input: AIMessage.
- Output: TextAccessor (in  the current LangChain version).

**When to use StrOutputParser:**  

- Chat applications.
- Content generation.
- Summaries.
- Explanations.
- Human-readable responses.

Limitation: produces unstructured text that is difficult for backend code to consume.  

---

### JsonOutputParser

Its purpose is to converts the LLM output into a python dictionary.

- Input: AIMessage containing valid JSON text.
- Output: `dict`.  

`get_format_instructions()` provides prompt instructions that encourage the LLM to produce valid JSON.  

Example: 

```text
Return a JSON object.
```

Important observation: 

- The parser does not force the model to return JSON.  
- It only provides instructions before generation.  
- After generation it attempts to parse the response.  
- If parsing fails, an exception is raised.  
- JsonOutputParser parse and validate valid JSON not valid schema.

**Valid JSON**  

Correct JSON syntax.

Example:

```json
{
  "country": "Egypt"
}
```

**Valid schema**  

JSON that follows a predefined structure.  

Example:

```json
{
  "country": "...",
  "capital": "...",
  "population": ...
}
```

A response may be valid JSON while still violating the expected schema.  

**When to use JsonOutputParser:**  

- Backend services.
- APIs.
- Automation.
- Data extraction.
- Pipelines.
- Intermediate processing.

**Common mistakes:**  

- Forgetting `get_format_instructions()`.
- Assuming valid JSON means the expected keys exist.
- Treating AIMessage as a dictionary.
- Expecting Output parsers to change the model's reasoning.  

## Lesson 4: Structured output with pydantic

Pydantic is a python library used to define, validate, and mange structured data.  
Instead of manually checking every field in a dictionary, we define teh expected structure once using a class.  

Without pydantic:

```python
if "name" is not in data:
  ...

```

With pydantic:

```python
class Country(BaseModel):
  name: str
  capital: str

```

Pydantic handles the validation of the attributes automatically.

### LangChain + pydantic flow

```text
BaseModel
     │
     ▼
Pydantic generates JSON Schema
     │
     ▼
PydanticOutputParser
     │
     ▼
get_format_instructions()
     │
     ▼
   Prompt
     │
     ▼
    LLM
     │
     ▼
    JSON
     │
     ▼
Validation
     │
     ▼
Pydantic Object
```

---

### BaseModel

`BaseModel` is the foundation of every pydantic model. It gives teh class the ability to:

- Validate data.
- Enforce data types.
- Generate JSON Schema.
- Serialize and deserialize data.
- Integrate with frameworks such as FastAPI and LangChain.

---

### JSON vs python dictionary

Python dictionary:

```python
{
    "name": "Egypt"
}
```

JSON:

```json
{
    "name": "Egypt"
}
```

They look very similar.

The difference:

- Dictionary → Python object in memory.
- JSON → Text used to exchange data which lives on the standard storage like hard drive.

---

### Validation

Pydantic validates data while creating the object.

Example:

```python

class Country(BaseModel):
    name: str
    capital: str

Country(
    name=123,
    capital="Cairo"
)
```

Result:

```text
ValidationError
```
If validation fails, the object is never created. This is called **Fail fast**  

### Field

`Field()` provides additional metadata for a field.  

```python
from pydantic import Field

class country(BaseModel):
  name: str = Field(
    description = "The common name of the country."
  )
```

The description is not mainly for python. It becomes part of teh generated schema used by LangChain.  

### PydanticOutputParser

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(
  pydantic_object = Country
)
```

unlike `JsonOutputParser`, this parser returns a pydantic object instead of dictionary.  

### get_format_instructions()

```python
parser.get_format_instructions()
```

Unlike `JsonOutputParser`, it generates a complete JSON schema containing:

- Model description.
- Field names.
- Field types.
- Required fields.
- Field descriptions.

This schema is inserted into the prompt to guide the LLM.

---

### JsonOutputParser vs PydanticOutputParser

**JsonOutputParser:**  

- Returns a python dictionary.
- Guarantees valid JSON.
- Does not validate the schema.
- Good for flexible structures.

**PydanticOutputParser:**

- Returns a pydantic object.
- Guarantees valid JSON.
- Validates against the schema.
- Raises `ValidationError` when data is invalid.
- Best for production systems.

### Choosing the right parser

Use `StrOutputParser` when:

- You only need plain text.

Use `JsonOutputParser` when:

- You need JSON.
- The structure is flexible or unknown.

Use `PydanticOutputParser` when:

- The backend expects a fixed schema.
- Validation is required.
- The output will be consumed programmatically.

### Key Takeaways

- `BaseModel` defines the expected data structure.
- `Field()` adds metadata that becomes part of the generated JSON Schema.
- `PydanticOutputParser` automatically converts LLM output into validated Python objects.
- Validation occurs after the LLM generates its response.
- Nested Pydantic models allow complex structured outputs.
- Use `JsonOutputParser` for flexible JSON.
- Use `PydanticOutputParser` for production-ready structured data.