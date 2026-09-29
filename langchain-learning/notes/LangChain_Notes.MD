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
  - [Lesson 5: LCEL (LangChain Expression Language)](#lesson-5-lcel-langchain-expression-language)
    - [Why LCEL exists?](#why-lcel-exists)
    - [Runnable](#runnable)
    - [Runnable sequence](#runnable-sequence)
    - [The pipe operator (`|`)](#the-pipe-operator-)
    - [invoke()](#invoke)
    - [batch()](#batch)
    - [stream()](#stream)
    - [ainvoke()](#ainvoke)
      - [Running `ainvoke()`](#running-ainvoke)
    - [Sync vs Async](#sync-vs-async)
    - [Production perspective: LCEL](#production-perspective-lcel)
    - [Key takeaways: LCEL](#key-takeaways-lcel)
  - [Lesson 6: Runnables](#lesson-6-runnables)
    - [Why Runnables exist?](#why-runnables-exist)
    - [The runnable interface](#the-runnable-interface)
    - [Composition](#composition)
    - [RunnableLambda](#runnablelambda)
    - [RunnablePassthrough](#runnablepassthrough)
    - [RunnableParallel](#runnableparallel)
    - [RunnableSequence](#runnablesequence)
    - [Building a RAG-shaped pipeline](#building-a-rag-shaped-pipeline)
    - [Composition vs Execution](#composition-vs-execution)
    - [Production perspective: Runnables](#production-perspective-runnables)
    - [Key Takeaways \& Engineering Practices: Runnables](#key-takeaways--engineering-practices-runnables)
      - [Key takeaways: Runnables](#key-takeaways-runnables)
      - [Common mistake: Runnables](#common-mistake-runnables)
      - [Best practice: Runnables](#best-practice-runnables)
    - [Personal observations: Runnables](#personal-observations-runnables)
  - [Lesson 7: Tools](#lesson-7-tools)
    - [Why tools exist?](#why-tools-exist)
    - [Tool call and Tool execution](#tool-call-and-tool-execution)
    - [Tool fundamentals](#tool-fundamentals)
      - [Tool definition](#tool-definition)
      - [Tool schema](#tool-schema)
      - [Pydantic and Enums](#pydantic-and-enums)
    - [Tool interaction](#tool-interaction)
      - [`bind_tools()`](#bind_tools)
      - [Tool calls vs. content](#tool-calls-vs-content)
      - [ToolMessage](#toolmessage)
      - [Tool descriptions](#tool-descriptions)
      - [Tool design as an interface](#tool-design-as-an-interface)
    - [Tool architecture](#tool-architecture)
      - [Production perspective: Tools](#production-perspective-tools)
      - [Tool vs RAG](#tool-vs-rag)
      - [Deterministic workflow vs. agent](#deterministic-workflow-vs-agent)
    - [Key Takeaways \& Engineering Practices: Tools](#key-takeaways--engineering-practices-tools)
      - [Common mistakes: Tools](#common-mistakes-tools)
      - [Best practices: Tools](#best-practices-tools)
      - [Key takeaways](#key-takeaways-1)
  - [Lesson 8: Retrievers](#lesson-8-retrievers)
    - [What is retrieval?](#what-is-retrieval)
    - [Document](#document)
    - [Retrieve as a runnable](#retrieve-as-a-runnable)
    - [Custom retrievers](#custom-retrievers)
    - [Vector retrieval architecture](#vector-retrieval-architecture)
      - [Embedding model](#embedding-model)
      - [Vector store](#vector-store)
      - [Retriever](#retriever)
    - [Vector store and retriever](#vector-store-and-retriever)
      - [Similarity search](#similarity-search)
      - [Top-k](#top-k)
      - [MMR](#mmr)
      - [Metadata filtering](#metadata-filtering)
    - [Retriever vs. vector store](#retriever-vs-vector-store)
    - [Retriever vs. RAGa](#retriever-vs-raga)
    - [When to use a retrieve](#when-to-use-a-retrieve)
    - [When retrieval alone is not enough](#when-retrieval-alone-is-not-enough)
    - [Key Takeaways \& Engineering Practices: Retrievers](#key-takeaways--engineering-practices-retrievers)
      - [Common mistakes: Retrievers](#common-mistakes-retrievers)
      - [Best practice: Retrievers](#best-practice-retrievers)
      - [Key takeaways: Retrievers](#key-takeaways-retrievers)
  - [Lesson 9: RAG (Retrieval augmented generation)](#lesson-9-rag-retrieval-augmented-generation)
    - [What is RAG?](#what-is-rag)
    - [Why RAG exists](#why-rag-exists)
    - [Grounding](#grounding)
    - [Retrieval quality](#retrieval-quality)
    - [Conversational RAG](#conversational-rag)
    - [RAG with LCEL](#rag-with-lcel)
    - [Manual RAG vs. LCEL RAG](#manual-rag-vs-lcel-rag)
    - [Production perspective: RAG](#production-perspective-rag)
    - [Key Takeaways \& Engineering Practices: RAG](#key-takeaways--engineering-practices-rag)
      - [Common mistakes: RAG](#common-mistakes-rag)
      - [Mental model: RAG](#mental-model-rag)
      - [Key takeaway: RAG](#key-takeaway-rag)
  - [Lesson 10: Agents](#lesson-10-agents)
    - [Deterministic workflow vs. agents](#deterministic-workflow-vs-agents)
    - [The agentic loop](#the-agentic-loop)
    - [Why agents exist](#why-agents-exist)
    - [Minimal agent](#minimal-agent)
    - [Tool calling vs agents](#tool-calling-vs-agents)
    - [Reading an agent execution trace](#reading-an-agent-execution-trace)
    - [Multiple tool calls](#multiple-tool-calls)
    - [Information Gaps and Tool Dependencies](#information-gaps-and-tool-dependencies)
    - [Tool Boundaries and Agent Decision-Making](#tool-boundaries-and-agent-decision-making)
    - [RAG vs agents](#rag-vs-agents)
    - [Production perspective: Agents](#production-perspective-agents)
    - [Key Takeaways \& Engineering Practices: Agents](#key-takeaways--engineering-practices-agents)
      - [Common mistakes: Agents](#common-mistakes-agents)
      - [Key mental models: Agents](#key-mental-models-agents)
      - [Personal observations: Agents](#personal-observations-agents)

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

## Lesson 5: LCEL (LangChain Expression Language)

LCEL (LangChain expression language) is LangChain's standard way of composing multiple runnables into single execution pipeline.  

Instead of manually invoking each component on e after another:

prompt -> LLM -> output parser  

LCEL allows us to express the same pipeline declaratively. The pipeline itself becomes anther runnable.

---

### Why LCEL exists?

Without LCEL, every step must be executed manually.  
Conceptually:

```text
Prompt.invoke()
    ↓
LLM.invoke()
    ↓
Parser.invoke()
```

As pipelines become larger, this approach becomes repetitive, harder to read, harder to maintain, and easier to make mistakes.  

LCEL removes the orchestration code while preserving exactly the same execution flow.  

The workflow does not changes.  

Only who performs the orchestration changes.  

Before:
> The developer orchestrate the pipeline.  

After:
> LangChain orchestrates the pipeline.

---

### Runnable

Any object that flows LangChain's standard execution interface.

Examples include:

- `ChatPromptTemplate`
- `ChatGoogleGenerativeAI`
- `StrOutputParser`
- `JsonOutputParser`
- `PydanticOutputParser`
- `RunnableSequence`

Because they all implement the same interface, they can connected together regardless of their internal implementation.  

This is one of the fundamental design ideas behind LangChain.  

### Runnable sequence

When multiple Runnables are connected using LCEL, LangChain creates a `RunnableSequence`.  

Conceptually prompt -> LLM -> parser becomes RunnableSequence. The sequence behaves exactly like any other runnable. It means is also supports:  

- `invoke()`
- `batch()`
- `stream()`
- `ainvoke`

So when you create a RunnableSequence called `chain` you can do runtime verification using:  
`type(chain)`
> langchain_core.runnables.base.RunnableSequence

---

### The pipe operator (`|`)

The pipe operator connects runnables together.

using:

```python
chain = prompt | llm | pydantic_parser
```

Conceptually: prompt -> LLM -> parser  

Each runnable automatically receives the output of the previous runnable. Nothing  is executed when the pipeline is built.  
Execution begins only when one of the execution methods is called.  

---

### invoke()

Process a single input synchronously.

Input: one input.  
output: one output.

Returns whatever the last runnables returns.

Examples:

- `AImessage`
- `TextAccessor`
- `dict`
- `student`
- Any custom pydantic model

### batch()

Processes multiple independent inputs.

Input: List[input]  
Output: List[output]  

Conceptually:  

```text
invoke()
  ↓
Output

becomes

batch()
  ↓
List[Output]
```

Internally, LangChain may execute requests concurrently depending on the runnable implementation.  

### stream()

Returns the output incrementally instead of waiting for the complete response.  

Conceptually:  
Instead of wait then complete response the user receives chunk -> chunk -> chunk.  
Streaming improves the perceived responsiveness of AI applications.  

Runtime observation (our environment):

- Each streamed chunk is a `TextAccessor`.
- Gemini streams text chunks rather than exposing individual tokens directly.

### ainvoke()

Asynchronous version of invoke()

Conceptually:  

`chain.invoke(...)` becomes await `chain.ainvoke(...)`  
The returned object is identical to invoke, only the execution strategy changes.  

#### Running `ainvoke()`  

**In Jupyter Notebook:**

```python
result = await chain.ainvoke(...)
```

**In a normal python script:**

```python
import asyncio 
result = asyncio.run(main())
```

where `main()` is an `async` def function.

---

### Sync vs Async

Async does not make the LLM faster.  

Instead, it allows your application to continue doing useful work while waiting for external services.

Example:

FastAPI sends a request to Gemini.  
While Gemini is generating a response:

- another HTTP request can be processed,
- another database query can run,
- another user can be served.

The application becomes more efficient because waiting time is no longer wasted.

I.e.
> Async does not make external services faster; it makes your application more efficient while waiting for them.

---

### Production perspective: LCEL

LCEL is the standard way production LangChain applications are built.

Typical use cases:

- `invoke()` → Single user requests.
- `batch()` → Offline processing and bulk inference.
- `stream()` → Chatbots and real-time AI interfaces.
- `ainvoke()` → FastAPI and other asynchronous web frameworks.

Almost every production AI application combines these execution methods depending on the workload.

### Key takeaways: LCEL

- LCEL is LangChain's  execution language.
- Every major LangChain component is a runnable.
- Multiple runnables compose into a RunnableSequence.
- RunnableSequence is itself a runnable.
- LCEL changes execution, not architecture.
- The pipeline always returns the output of its last runnable.
- Different execution methods exist fro different application requirements.
- Async improves application scalability, not model speed.

## Lesson 6: Runnables

A Runnable is a LangChain abstraction representing a unit of computation that follows a common execution interface.

Runnables can represent many different things:

- Prompt templates.
- LLMs.
- Output parsers.
- Python functions.
- Parallel branches.
- Complete chains.

The important idea is that different concrete classes can follow the same Runnable interface.  
This allows LangChain to treat them uniformly.  

The common execution interface includes methods such as:

- `invoke()`
- `batch()`
- `stream()`
- `ainvoke()`

The exact output depends on what the Runnable does.

### Why Runnables exist?

Without a common abstraction, every LangChain component could have completely different ways of being executed and composed.  

For example:

```text
Prompt → some prompt-specific API

LLM → some LLM-specific API

Parser → some parser-specific API
```

This would make composition difficult.

Runnables provide a common contract:

```text
Different concrete components
          ↓
    Runnable interface
          ↓
Can be composed and executed uniformly
```

This allows LangChain to build larger systems from smaller components.

---

### The runnable interface

A prompt, LLM, parser, and RUnnable sequence are different concrete types.  
They are not all literally the same python class.  
Instead, they conform to the Runnable abstraction.
This distinction is very important:

```text
Concrete type ≠ Interface
```

For example:

```text
ChatPromptTemplate
ChatGoogleGenerativeAI
StrOutputParser
RunnableLambda
RunnableParallel
RunnableSequence
```

Can all participate in the Runnable ecosystem.  
The caller does not need to know their internal implementation in order to execute them through the common interface.

---

### Composition

A major benefit of the Runnable abstraction is the composition.  
A Runnable can be combined with another Runnable:

```text
Runnable A
    ↓
Runnable B
    ↓
Runnable C
```

The output of one Runnable becomes the input of the next.

For example:

```text
Prompt
  ↓
LLM
  ↓
Parser
```

can be composed using LCEL:

```python
chain = prompt | llm | parser
```

The resulting object is a `RUnnableSequence` which is a Runnable itself.  
Therefore, an entire pipeline can be treated as one component and composed into a larger pipeline. This allows a chain of small Runnables to create Runnable sequence.  

This is the basis of reusable and modular pipelines.

---

### RunnableLambda

`RunnableLambda` adapts an ordinary python callable into a Runnable.

For example:

```python
def double(x):
    return x * 2
```

is an ordinary Python function.

Wrapping it:

```python
double_runnable = RunnableLambda(double)
```

turns it into a Runnable.

It can now participate in LCEL:

```python
Python function
      ↓
RunnableLambda
      ↓
Runnable
      ↓
LCEL pipeline
```

The function behavior does not change. The wrapper gives it the Runnable interface.

This is useful as it allows application-specific python logic to be inserted into LangChain pipelines without creating a custom Runnable class for every small operation,

Examples include:

- Cleaning text.
- Formatting data.
- Extracting information from objects.
- Transforming model output.
- APPlying custom business logic.

**Input and Output**  

`RunnableLambda` does not impose a particular input or output type.  
The wrapped function determine them.  
Example:

`dict → str` or, `int → int` or, `Pydantic object → str`.

The only requirement for composition is that  the output of the previous Runnable is compatible with the function's expected input.

### RunnablePassthrough

`RunnablePassthrough` receives an input and returns that same input unchanged.

Conceptually:

```text
Input
  ↓
RunnablePassthrough
  ↓
Same Input
```

The input remains the same in the same format after passing through the Runnable.

It is usefulness becomes apparent when data needs to be preserved while other branches process the same input.

For example:

```text
                 ┌──→ original question
Input ───────────┤
                 └──→ retriever → context
```

The original question can be preserved while another Runnable process it.

This makes `RunnablePassthrough` especially useful in RAG-style pipelines.

### RunnableParallel

`RunnableParallel` sends the same input through multiple Runnable branches, and their output are combined into one result.

Conceptually:

```text
                 ┌──→ Runnable A
Input ───────────┤
                 └──→ Runnable B
```

The results are combined into a dictionary.

For example:

```text
Input: 5

        ┌──→ Passthrough → 5
        │
        ├──→ double → 10
        │
        └──→ add_ten → 15

Result:
{
    "original": 5,
    "doubled": 10,
    "plus_ten": 15
}
```

**Important distinction**  
RunnableParallel means that multiple Runnable branches receive input.  
It should not automatically be interpreted as python multiprocessing or CPU parallelism.  
The concept here is primarily parallel data flow / branching.  
The actual execution and concurrency behavior depends on the implementation and runtime.  

### RunnableSequence

`RunnableSequence` represents sequential composition.  

For example: `A -> B -> C`  

When we write:

```python
chain = prompt | llm | parser
```

LangChain creates a `RunnableSequence`.  
When the sequence is executed:

```python
chain.invoke(...)
```

The data flows through the Runnables in order:

```text
Input
  ↓
prompt.invoke()
  ↓
PromptValue
  ↓
llm.invoke()
  ↓
AIMessage
  ↓
parser.invoke()
  ↓
Final result
```

The intermediate steps are orchestrated automatically.  
The developer does not need to manually invoke every Runnable.  

**RunnableSequence is also a Runnable**  

This is one of the most important concepts of the lesson.  
A sequence can itself be used as a Runnable:

```text
Runnable A
   ↓
Runnable B
   ↓
RunnableSequence
   ↓
Another Runnable
```

Therefore, smaller pipelines can be composed into larger pipelines.  
This provides modularity and reusability.

---

### Building a RAG-shaped pipeline

The notebook demonstrated how these Runnables types can be combined into a simplified RAG-shaped pipeline.  

The architecture is:

```text
                    ┌──→ question ───────────┐
User Question ──────┤                         ↓
                    └──→ Retriever → context → Prompt
                                                  ↓
                                                 LLM
                                                  ↓
                                                Parser
                                                  ↓
                                               Answer
```

The data-preparation stage can use:

```text
RunnablePassthrough
        +
RunnableLambda / Retriever
        ↓
RunnableParallel
```

Which produces:  

```text
{
    "question": ...,
    "context": ...
}
```

That output can then be passed into a `ChatPromptTemplate`.  
The prompt produces a `ChatPromptValue`, which can be passed to the LLM.  
The LLM produces an `AIMessage`, which can then be passed to an output parser.  
The final result is therefore a composition of several different Runnables.  

The important lesson is not the fake retriever used in the notebook but the important architectural pattern is:  

```text
Prepare data
     ↓
Build prompt
     ↓
Generate
     ↓
Parse
```

Each stage can be independently replaced or extended because each stage follows the Runnable abstraction.

---

### Composition vs Execution

This distinction is critical.  
The pipe operator:

```python
chain = prompt | llm | parser
```

composes Runnables. It does not execute them.  
No request is sent to the LLM simply because the chain was created.  
Execution happens only when an execution method is called as `chain.invoke(...)` or `chain.batch(...)`  or `chain.stream(...)`  or `chain.ainvoke(...)`  

Therefore, the pipe operator `|` is Composition while the Execution is done by the execution methods.  

This distinction makes LCEL much easier to reason about

### Production perspective: Runnables

Runnables provide a modular way to construct real AI application pipelines.  
A production pipeline might conceptually look like:

```text
User Input
    ↓
Input Validation
    ↓
Parallel Data Preparation
    ↓
Retriever
    ↓
Prompt Construction
    ↓
LLM
    ↓
Structured Output Parser
    ↓
Business Logic
    ↓
API Response
```

Each stage can be represented by a Runnable.  
This allows individual components to be:

- Replaces.
- Tested.
- Reused.
- Composed.
- Extended.

Runnables therefore provide an important architecture foundation for large LangChain applications.

### Key Takeaways & Engineering Practices: Runnables

#### Key takeaways: Runnables

- A Runnable is a com0posable unit of computation with a common execution interface.
- Different concrete classes can conform to the Runnable abstraction.
- Runnables support common execution methods such as `invoke()`, `batch()`, `stream()`, and `ainvoke()`.
- `RunnableLambda` adapts ordinary python functions into Runnables.
- `RunnableLPassthrough` preserves an input unchanged.
- `RunnableParallel` sends one input through multiple Runnables branches.
- `RunnableSequence` executes Runnables sequentially.
- A `RunnableSequence` is itself a Runnable.
- Small Runnable can therefore be composed into lager Runnables.
- The `|` operator composes Runnables; it does not execute them.
- Input/output compatibility is required for meaningful composition.
- The last Runnable determines the final output produced by the chain.

#### Common mistake: Runnables

- **Thinking all Runnables are the same class**: They are not, they can be different concrete classes while following the same Runnable abstraction.
- **Thinking `|` executes the pipeline**: it doesn't. It creates a composition. Execution happens later through methods such as `invoke()`.
- **Thinking Runnables means the pipeline will automatically work**: A component being a Runnable does not guarantee that its input is compatible with the previous component's output. Both the Runnable contract and compatible data flow matter.
- **Thinking `RunnableParallel` means multiprocessing**: It primarily represents multiple branches receiving the same input. Do not confuse this abstraction with python's deeper concurrency and parallelism mechanisms.
- **Thinking `RunnableLambda` changes the function**: It does'nt, it adapts the function to the Runnable interface.
- **Thinking `stream()` always splits output into many chunks**: It doesn't, th chunking behavior depends on the Runnable. Our `RunnableLambda(double)` example returned `10` as a single chunk

#### Best practice: Runnables

- Keep Runnables focused on clear responsibility.
- Build larger pipelines by composing smaller reusable components.
- Inspect intermediate outputs when debugging complex chains.
- Keep data structures between Runnables explicit and compatible.
- Use `RunnableLambda` for small pieces of custom python logic.
- Use `RunnableParallel` when multiple pieces of information need to be produced from the same input.
- Avoid creating unnecessarily large monolithic chains.
- Separate data preparation, generation, parsing, and business logic when possible.

### Personal observations: Runnables

The mental model during the lesson:

```text
Runnable
    ↓
Composable unit
    ↓
Can be combined with other Runnables
    ↓
Can become a RunnableSequence
    ↓
The sequence is still a Runnable
    ↓
The sequence can be composed again
```

A particularly important realization was that a complete chain can be treated as black box:

```text
Input -> RunnableSequence -> output
```

The caller does not need to know how many RUnnables exist internally.  
This makes it possible to build larger systems from smaller, reusable pipelines components.  

The lesson 6 notebook also verified the behavior of `RunnableLambda`, `RunnablePassthrough`, `RunnableParallel`, and `RunnableSequence` using the installed LangChain version.

## Lesson 7: Tools

A LangChain Tool is a controlled application capability exposed to an LLM through a structured interface.

A tool typically provides:

- Name.
- Description.
- Input schema.
- Execution logic.

The LLM can decide to request a tool, but the tool is actually executed by the application/runtime.

> A Tool is not simply "information given to the LLM." It is a capability the LLM can request the application to execute.

### Why tools exist?

An LLM cannot directly access arbitrary external systems or perform application operations.  
Tools provide a controlled interface between the model and external capabilities such as:

- Databases.
- APIs.
- Web search.
- Calculations.
- Filesystems.
- Business systems.
- Application actions.

Without tools the user directly got the answer from the LLM (User → LLM → Answer).

With tools the user ask the LLM to perform a task, if the LLM determines that a tool is needed it make a tool call, the application executes the requested tool then return the tool result back to the LLM, the LLM uses the tool result to generate the final answer.

workflow:

```text
User
 ↓
LLM
 ↓
Tool call
 ↓
Application executes Tool
 ↓
Tool result
 ↓
LLM
 ↓
Final answer
```

---

### Tool call and Tool execution

**The tool call does not mean tool execution.**  

Tool call is the models structured request:

```python
{
    "name": "get_customer_field",
    "args": {
        "customer_id": 4821,
        "field": "balance"
    }
}
```

Nothing has been executed yet.

Tool execution happens when the application invokes the tool.

```python
get_customer_field.invoke({
    "customer_id": 4821,
    "field": "balance"
})
```

The tool calling lifecycle:

```text
User
 ↓
Gemini
 ↓
AIMessage
 └── tool_calls[]
       ↓
Application executes Tool
       ↓
Tool result
       ↓
ToolMessage
       ↓
Gemini
 ↓
Final AIMessage
```

LLM responsibility is to decide if it needs a tool or not, and generating a tool call.

Application/orchestrator responsibility is to receive the tool call, execute the tool and return the tool result.

---

### Tool fundamentals

#### Tool definition

A python function can be converted into LangChain tool using the official `@tool` decorator.

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    ...
```

The function's type hints can be used to infer the Tool's input schema.

important tool properties:

```python
tool.name
tool.description
tool.args
```

`args` describes the Tool's expected input arguments, not its output.

#### Tool schema

simple type hints can create  a basic input schema:

```python
customer_id: int
```

which becomes conceptually: customer_id --> integer.

for more structured requirements, an explicit pydantic schema can be used.

For example:

customer_id --> integer.  
field --> CustomerField.

where:

```text
CustomerField
├── name
├── balance
└── status
```

This allows the tool interface to express actual constraints instead of merely documenting them.

#### Pydantic and Enums

Field(description=...) provides descriptive information but does not restrict a string to specific values.

An `Enum` represents a closed set of allowed values.

```text
field: str
        ↓
any string

field: CustomerField
        ↓
name | balance | status
```

This can then be used inside a pydantic `BaseModel` and supplied as the tool's `args_schema`.

---

### Tool interaction

#### `bind_tools()`

`bind_tools()` makes tools available to the model

```python
llm_with_tools = llm.bind_tools([get_customer_field])
```

This does not execute the tool and does not call Gemini (the LLM) immediately.  
It creates a configured model runnable that knows which tools are available when invoked.

Execution occurs latter:

```python
response = llm_with_tools.invoke(...)
```

#### Tool calls vs. content

The model still returns an `AIMessage`.  
If the model produces a normal textual response:

```python
response.content
```

is the relevant output.

If the model requests tools:

```python
response.tool_calls
```

contains the requested operations.

Conceptually: 

```text
AIMessage
│
├── content
│     → textual answer
│
└── tool_calls
      → requested operations
```

A model with tools bounded can still answer directly without using a tool.

#### ToolMessage

After the application executes a tool, its result can be represented as `ToolMessage`.  
The `tool_call_id` associates teh result with the exact tool call that produced it.

```text
AIMessage
 └── tool_call ID: call_123
          ↓
      Tool execution
          ↓
ToolMessage
 └── tool_call_id: call_123
```

This becomes especially important when multiple tool calls are involved.

#### Tool descriptions

Tool descriptions are part of the model-facing interface.  

A weak description: "Get customer information."  

A stronger description explains:

- What the tool does.
- When it should be used.
- Sometimes when it should not be used.

The tool description helps the model select the appropriate capability.  

#### Tool design as an interface

A tool has 2 consumers, the application which has access to the actual python implementation and execute the tool; and the LLM which selects the tool and provides the argument.

The LLM consumes the tool's description/schema, while the application consume the tool's implementation.  
The LLM choose the tool and generates the argument; the application executes the tool.

```text
                 Tool
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
   Application             LLM
        │                   │
 execution logic      selection + arguments
```

The application cares about correctness, validation, authorization,error handling and reliability.

The LLM additionally needs to understand what the capability does and how to request it.

---

### Tool architecture

#### Production perspective: Tools

A tool is a boundary between probabilistic model behavior and deterministic application code.  
A production tool should not blindly trust the model.  
A useful concept architecture is:

```text
LLM
 ↓
Tool schema
 ↓
Input validation
 ↓
Authorization / business rules
 ↓
Tool execution
 ↓
External system
```

The LLM decide which tool to use and generates its arguments. The application validates those arguments, applies authorization and business rules, then executes the tool to interact with the external system.

The important idea is that the LLM does not directly access the external system. THe application controls the execution path.

For failures:

```text
Internal exception
 ↓
Application error handling
 ↓
Controlled Tool-level error
 ↓
ToolMessage / runtime
 ↓
LLM
 ↓
User-facing response
```

If the tool execution fails, the application catches the internal exception and converts it into a controlled tool level error. That error is returned to the LLM, allowing it to understand the failure and produce an appropriate response to the user.

The key idea is the internal implementation errors should not blindly leak to the LLM or user; the application controls how failures are represented and communicated.

We discussed the production layers conceptually but did not fully implement them. Robust authorization, error handling, retries, observability, approval gets , and more advanced execution control will be developed later.

#### Tool vs RAG

Rag primarily retrieved relevant information and puts it into the model's context:


```text
Question
 ↓
Retriever
 ↓
Relevant documents
 ↓
LLM
```

Tools expose capabilities:

```text
Question
 ↓
LLM decides
 ↓
Tool
 ↓
External operation
 ↓
Result
 ↓
LLM
```

They can also be combined in the same system.

#### Deterministic workflow vs. agent

A deterministic workflow has a developer-defined execution path

Example having 3 functions/tools (`search_customer()`, `get_customer_orders()`,`send_email()`) and the system always executes them in specific order:

```text
search_customer
      ↓
get_customer_orders
      ↓
send_email
```

An agent can dynamically decide what action to take next:

```text
Goal
 ↓
LLM
 ├── Tool A
 ├── Tool B
 ├── Tool C
 └── Stop
      ↓
   Result
      ↓
     LLM
      ↓
   Next decision
```

An agent is not simply "an LLM with tools." The important property is dynamic decision-making over actions toward a goal.

---

### Key Takeaways & Engineering Practices: Tools

#### Common mistakes: Tools

- Assuming `bind_tools()` executes a tool.
- Assuming a tool call means the tool has already executed.
- Confusing tool input schema with tool output.
- Treating tool descriptions as developer-only document.
- Assuming the model must use a tool just because on is available.
- Trusting the LLM alone to enforce business or security rules.
- Using overly vague tool descriptions.
- Replacing deterministic workflows with agents when the workflow is already known.

#### Best practices: Tools

- Give tools narrow, well-defined responsibilities.
- Use accurate type hints.
- Use structured schemas when inputs require meaningful constraints.
- Write tool descriptions specifically for model tool selection.
- Validate tool inputs.
- Enforce authorization and business rules outside the model.
- Handle tool failures explicitly.
- Keep dangerous side-effecting tools behind appropriate controls or approval mechanisms.
- Prefer deterministic workflows when the execution path is known.
- Use agents when dynamic decision-making genuinely provides value.

#### Key takeaways

```text
Tool
→ controlled application capability

bind_tools()
→ makes Tools available to the model

tool_call
→ model requests an operation

Tool execution
→ application actually performs the operation

ToolMessage
→ communicates the Tool result back to the model

Agent
→ dynamically decides which actions to take and potentially what to do next
```

## Lesson 8: Retrievers

### What is retrieval?

Retrieval is the process of finding information relevant to a given query.

A Retriever is the LangChain abstraction responsible for retrieving relevant `Document` objects based on input query.

```text
Query: str 
    ↓ 
Retriever 
    ↓ 
List[Document]
```

Retrieval does not generate an answer.
Retrieval finds relevant information.

RAG retrieves relevant information, provides it to an LLM, and generates an answer.

---

### Document

LangChain represents retrieved information using the `Document` object.  
A `Document` contains two important components.

```text
Document 
├── page_content 
└── metadata
```

`Page_content` contains the actual text.  
`metadata` contains information describing the document, such as:

- Source.
- Page number.
- Document ID.
- Other application-specific information.

Metadata is useful for:

- Debugging retrieval.
- Source attribution.
- Filtering.
- Post-processing.
- Providing additional context.

Example

```python
Document(
  page_content="LangGraph is designed for stateful agent workflows.", metadata={"source": "langgraph.txt"}
)
```

---

### Retrieve as a runnable

`BaseRetrieve` is part of LangChain's runnable ecosystem.  
A retriever therefore supports the runnable interface:  

```python
retriever.invoke(query)
```

A custom retriever can implement:  

```python
_get_relevant_document_(query)
```

While `BaseRetriever` provides the surrounding runnable behavior.

Conceptually:

```text
retriever.invoke(query) 
    ↓ 
BaseRetriever
    ↓ 
_get_relevant_documents(query)
    ↓ 
List[Document]
```

This means the same runnable concepts learned previously such as `invoke()` and `batch()` also apply to retrievers.

`invoke()` receives one input query and return one output. The output can contain multiple documents.

`batch()` receives multiple inputs and returns multiple outputs.

### Custom retrievers

A custom retriever can implement retrieval logic appropriate for the application.

For example:

- Using metadata matching.
- Using Keyword search.
- Using embedding then vector similarity search

For return the relevant documents.  
The retriever abstraction does not require one specific retrieval algorithm.

### Vector retrieval architecture

A common semantic retrieval architecture is:

At indexing

```text
Documents 
    ↓ 
Embedding Model 
    ↓ 
Document Embeddings 
    ↓ 
Vector Store
```

At query time:

```text
User Query 
    ↓ 
Embedding Model 
    ↓ 
Query Embedding 
    ↓ 
Vector Store 
    ↓ 
Similarity Search 
    ↓ 
Relevant Documents
```

#### Embedding model

Converts text into numerical vectors representing the semantic meaning of the text.  
Both documents and queries need to be embedded when using vector similarity search.

#### Vector store

Store document embeddings and associated documents.  
It can perform vector similarity search against a query embedding.

#### Retriever

Provides the retrieval interface and uses the underlying retrieval mechanism to return relevant `Document` objects.


Embedding model: convert text into vectors.

Vector store: stores/searches the vectors.

Retriever: exposes retrieval as standard interface.

The embedding model and vector store are not themselves the definition of a retriever.

---

### Vector store and retriever

A vector store can expose a retriever:

```python
retriever = vector_store.as_retriever()
```

The application can then use:

```python
documents = retriever.invoke(query) 
```

This separates the application-facing retrieval interface from the underlying search implementation.

```text
Vector Store 
    ↓ 
.as_retriever() 
    ↓ 
Retriever 
    ↓ 
invoke(query) 
    ↓ 
List[Document]
```

#### Similarity search

Similarity search retrieves documents whose embeddings are most similar to the query embedding.

For example:

```text
Query: 
"How do I build stateful agent workflows?" 
    ↓ 
Query embedding 
    ↓ 
Similarity search 
    ↓ 
LangGraph document
```

Semantic search does not require the query to contain the exact words appearing in the document.  
The goal is to find documents that are most relevant according to the retrieval method.

#### Top-k

The `k` parameter controls how many final documents are returned.

Example:

```Python
retriever = vector_store.as_retriever( search_kwargs={"k": 2} )
```

`k = 2` does not mean that those 2 documents contain the answer, it means return the 2 documents judged most relevant by the configured retrieval method.  
Retrieval is therefore a candidate-selection stage, not an oracle.

#### MMR

MMR stands for maximal marginal relevance.  

Normal similarity search focuses primarily on relevance.  
MMR balance relevance to query and diversity among selected documents.

This helps avoid returning several nearly identical documents or overlapping chunks.

Example:

```text
Similarity search: 
A — Remote employees receive 21 vacation days. 
B — Remote employees receive 21 vacation days per year. 
C — Remote employees follow the same vacation policy.
```

MMR may prefer:

```text
A — vacation days 
D — request procedure 
E — carry-over rules
```

Because these provide more diverse information.  
MMR does not ignore relevance. it considers both relevance and redundancy.

A retriever can be configured to use MMR:

```python
retriever = vector_store.as_retriever( 
  search_type="mmr",
  search_kwargs={ 
    "k": 3, 
    "fetch_k": 10 
    }
  )
```

`fetch_k`: candidate pool consider by MMR.  
`k`: final number of documents returned.  

`lambda_mult` controls the balance between the relevance and diversity.  
Higher emphasis on relevance makes behavior more similar to ordinary similarity search, while greater emphasis on diversity encourages less results.

#### Metadata filtering

Metadata can also be used to restrict the retrieval candidate pool.

For example:  
Having 1,000,000 documents, `filter: topic = "agents"` so it filters the eligible documents into 2000 eligible documents, and after applying semantic search with `k = 5` we got hte most relevant documents where topic is agents.

Filtering and semantic search solve different problems:  
Metadata filtering asks "Which documents are eligible?" while semantic search asks "Which eligible documents are most relevant?".

In a production system, metadata filtering can be especially important for:

- Tenant isolation.
- Document categories.
- Permissions.
- Source restrictions.
- Dates.
- Departments.

However, authorization should not be delegated to the LLM. Security and access-control rules should be enforced by the application/data layer.

---

### Retriever vs. vector store

Vector store: stores and searches vectors.  
Retriever: provides a standardized retrieval interface.

A retriever may use a vector store internally, but retrieval does not inherently require vector search.

Other retrieval strategies include:

- Keyword/lexical search.
- Database queries.
- Metadata filtering.
- Hybrid search.
- Other domain-specific retrieval methods.

### Retriever vs. RAGa

Retriever workflow:

```text
Query 
  ↓ 
Retriever 
  ↓ 
Relevant Documents
```

RAG workflow:

```text
User Query 
  ↓ 
Retriever 
  ↓ 
Relevant Documents 
  ↓ 
Context / Prompt 
  ↓ 
LLM 
  ↓ 
Generated Answer
```

The retriever finds the evidence, while the LLM generates the answer. RAG combines both stages.

---

### When to use a retrieve

Use a retriever when an application needs to dynamically find relevant information from a collection of data.

Common examples:

- RAG systems.
- Enterprise knowledge bases.
- Document search.
- Customer-support systems.
- Question answering.
- Coding assistants.
- Agent Workflows requiring external knowledge.

### When retrieval alone is not enough

Retrieval does not generate an answer.  
If the application needs a natural-language response based on retrieved information, retrieval must combined with a generation component such as an LLM.

Also, retriever quality depends heavily on the underlying retrieval strategy and data preparation.  
Poor retrieval can produce poor RAG even when the LLM itself is capable.

---

### Key Takeaways & Engineering Practices: Retrievers

#### Common mistakes: Retrievers

Confusing retrieval with RAG:  
Retrieval only finds relevant information, RAG combines retrieval with generation.

Assuming retrievers require embeddings:  
Embeddings are common for semantic/vector retrieval but are not required by the retriever abstraction.

Assuming k means correct answers:  
`k`  controls the number of returned candidates, not whether those documents actually contain the answer.

Confusing vector stores with retrievers:  
A vector store performs storage/search operations.  
A retriever exposes a retrieval interface.

Assuming more documents retrieving always means better RAG:  
Increasing `k` can introduce irrelevant or redundant information and increase context size, latency, and cost.

Assuming MMR simply picks different words:  
MMR considers similarity between documents semantically; it is not merely checking whether documents share the same literal words.

#### Best practice: Retrievers

- Treat retrieval as an independent engineering stage.
- Preserve useful metadata with every `Document`.
- Tune `k` according to the application rather than choosing it arbitrarily.
- Consider MMR when retrieved chunks are highly redundant.
- Use metadata filtering to constrain the retrieval candidate pool when appropriate.
- Evaluate retrieval quality separately from generation quality.
- Keep authorization and security enforcement outside the LLM.
- Remember that retriever configuration and filter syntax can depend on the underlying vector store.

#### Key takeaways: Retrievers

Retriever: take a query input and return `list[Document]` output.

Document: consist of page_content + metadata.

Embedding model: convert text to vector.

Vector store: Stores and searches vectors.

Similarity: prioritize relevance.

MMR: combine relevance with diversity.

`K`: number of final results.

Metadata filter: restrict eligible documents.

Most importantly:  
> A retriever is an abstraction for retrieving relevant documents. Vector search is one implementation of retrieval, not the definition of retrieval.  
Retrieval finds the evidence; RAG uses that evidence to generate an answer.

## Lesson 9: RAG (Retrieval augmented generation)

### What is RAG?

RAG (Retrieval augmented generation) is an application architecture that retrieves relevant external information and provides it to an LLM as context before generation.

basic work flow:

```text
User Question
      ↓
Retriever
      ↓
Relevant Documents
      ↓
Context
      ↓
Prompt
      ↓
LLM
      ↓
Answer
```

RAG is not a special type of LLM. It is a system architecture built around an LLM.

### Why RAG exists

LLMs have limitations:

- Their knowledge may be outdated.
- They may not have access to private organizational data.
- They can hallucinate when information is unavailable.
- Putting an entire knowledge base directly into the prompt is inefficient.
- Fine-tuning is not the appropriate solution for every knowledge-retrieval problem.

RAG addresses this by retrieving relevant information at query time and giving it to the model as context.

Instead of asking LLM "Answer this from everything you know."

We build RAG system where the relevant information is retrieved fro the knowledge base based on the question then both the question and the knowledge information are given to the LLM, the LLM then generate a grounded answer:

```text
Question
   ↓
Retrieve relevant knowledge
   ↓
Give that knowledge to the LLM
   ↓
Generate a grounded answer
```

**What is the Engineering purpose of RAG?**  
The primary purpose of RAG is to connect an LLM to external knowledge at inference time.

**When Should You Use RAG?**  
RAG is particularly useful when an application needs to answer question using external or changing information.  

Typical sources includes:

- **Applications:**
  - Documentation assistants
  - Enterprise knowledge bases
  - Customer-support systems
  - Document question answering
  - Research assistants
  - Internal company search
  - Applications over private data

- **Data sources:**
  - Company documents
  - PDFs
  - Databases
  - Documentation
  - Knowledge bases
  - Product information
  - Internal policies
  - Search indexes

RAG is less useful when the task does not require external knowledge or retrieval.

RAG separate two responsibilities:

Retrieval → Find useful information.  
Generation → Produce a useful response.  

The retriever does not generate the answer.  
The LLM does not need to know the entire knowledge base beforehand.

---

### Grounding

A RAG system should generally instruct the LLM to use only teh supplied context when the application requires grounded answers.

Example:

```text
Answer the question using ONLY the provided context.  

If the context does not contain enough information to answer the question, say that you don't have enough information.
```

This creates a **knowledge boundary** around the answer.

However, this is not a complete guarantee against hallucination.  

If retrieval fails to find information that actually exists in the knowledge base, the LLM may incorrectly conclude that the information is unavailable.  
Therefore good grounding + good retrieval = More reliable RAG.

Production RAG may additionally use:

- Retrieval thresholds.
- Better chunking.
- Query rewriting.
- Re-ranking.
- Hybrid retrieval.
- Metadata filtering.
- Evaluation.
- Citation/Source tracking.

### Retrieval quality

The objective of RAG is to retrieve the smallest useful set of high-quality evidence/documents, **not** retrieving as many documents as possible.  

Too few documents can cause missing information.  
While too many documents can cause:

- Irrelevant contexts.
- Higher token usage.
- Higher latency.
- More difficult reasoning.
- context-window pressure.

The goal is therefore relevant context, not maximum context.

**Retrieval quality vs. generation quality**  

RAG introduces two major quality dimensions.  
Retrieval quality: did we retrieve the right information?  
Generation quality: did the LLM correctly use that information?

These problems should be considered separately.  

A system can have excellent generation but poor retrieval. It can also retrieve excellent documents but produce a poor answer because the generation step is badly designed.  
This separation is useful when debugging RAG systems.

If the answer is wrong ask: was the information missing from the retrieved context?  
If yes, investigate the retrieval. If the correct information was present but the model ignored or misunderstood it, investigate generation.

---

### Conversational RAG

Normal RAG treats the current query as the retrieval input.  
This becomes problematic when the user asks flow-up questions.  
For example:

> User: What is LangGraph?

Then:

> User: How is it different from LangChain?

The second question depends on the conversation. A retriever receiving only "How is it different from LangChain?".

So it may not have enough information to understand what "it" refers to.

Conversational RAG therefore introduce conversation history into the retrieval process.  
The important distinction is that conversation  history provides conversational context. Retrieved documents provide knowledge context.

They serve different purposes and should not automatically be treated as the same thing.

**Quey rewriting**  

One common solution to conversational retrieval is query rewriting.  
The system use conversation history and the user's  latest question to produce a standalone query.

Example:
Conversation: What is LangGraph?  

Follow-up: How is it different from LangChian?

Rewritten retrieval query: How is LangGraph different LangChian?

The rewritten query is then sent to the retriever.  

An important architectural distinction is that the rewritten query is primarily a retrieval query.  
The original use question should normally remain the question used for the final response.  
This prevents the retrieval-oriented transformation from unnecessarily changing what user actually asked.

**Conversation history is not retrieved context**  

These two concepts are easy to mix up

**Conversation history:** contains previous interactions between the user and the assistant.

**Retrieved context:** contains information retrieved from external knowledge base for the current question.

History:

> Human what LangGraph?  
> AI: LangGraph is designed for stateful agent workflows.

Retrieved context:

> LangGraph is a framework for building stateful agent workflows...

The history tells the system what has been discussed.  
The retrieved context provides external knowledge relevant to the current question.

In conversational RAG, the history can help construct a better retrieval query, while the newly retrieved documents provide the knowledge used for the answer.

### RAG with LCEL

LCEL allows the RAG architecture to be represented as a composable pipeline.

The important idea is not memorizing the exact chain syntax, but understanding the data flow.

A conversational RAG pipeline can conceptually be viewed as:

```text
Input
  ↓
Conversation-Aware Query Rewriting
  ↓
Retrieval
  ↓
Document Formatting
  ↓
Combine:
  ├── Original Question
  ├── Conversation History
  └── Retrieved Context
  ↓
Prompt
  ↓
LLM
  ↓
Output Parser
  ↓
Answer
```

LCEL provides the composition mechanism that connects these stages.

Useful runnable concepts encountered here include:

- **`RunnableParallel`:** Runs multiple branches from the same input and combines their results into a dictionary. This useful when several pieces of information need to be prepared independently.

- **`RunnableLambda`:** Turns a normal python callable into a Runnable. It is useful for small transformation, such as extracting one field from an input dictionary or formatting data.

- **`RunnablePassThrough`:** Passes the input through unchanged. It is useful when a value needs to remain available while other branches transform the same input

Key architecture:

> Use a RUnnable because it represent a meaningful transformation or routing step, not simply because everything could theoretically be wrapped in one.

### Manual RAG vs. LCEL RAG

A manually implemented RAG makes the individual operations explicit(صريحة). This is useful for learning because you can clearly see query rewriting, retrieval, context construction, prompt construction, and generation.  

LCEL expresses the same architecture as composable pipeline. The advantage is not that LCEL magically makes RAG work better.  
Its advantage is that the pipeline becomes easier to compose, reuse, inspect, and extend.  
The underlying RAG architecture remains the same.

Manual implementation teaches the mechanism.  
LCEL provides a structured way to compose those mechanics.

### Production perspective: RAG

A production RAG system is considerable more sophisticated than the basic pipeline studied here

Real system may add:

- Better document parsing.
- Intelligent chunking.
- Hybrid search.
- Query rewriting.
- Re-ranking
- Retrieval thresholds.
- Deduplication.
- Contextual compression.
- Caching.
- Evaluation pipelines.
- Observability.
- Access control
- source citations.

The basic RAG pipeline should therefore be viewed as the foundation rather than the final architecture.

Useful production model is:  
Ingestion → Indexing → Retrieval → Context Selection → Generation → Evaluation

Each stage can become its own engineering problem

---

### Key Takeaways & Engineering Practices: RAG

#### Common mistakes: RAG

**Retrieving too many documents**  
  More context can mean more noise rather than more knowledge.  

**Assuming similarity means relevance**  
 An embedding similarity score is not the same thing as semantic correctness

**Ignoring retrieval failures**  
  A wrong answer may originate from the retriever rather that the LLM.

**Treating conversation history as the knowledge base**  
  History  helps understanding the conversation retrieved documents provide external knowledge.

**Using the rewritten query as the final user question**  
  Query rewriting is primarily a retrieval operation. The original question should generally remain the user-facing question.

#### Mental model: RAG

The most important mental model from this lesson is: The application retrieves information first and places that information into the model's context.

Therefore:

```text
User question
     ↓
Understand the question
     ↓
Formulate an effective retrieval query
     ↓
Retrieve relevant information
     ↓
Select useful context
     ↓
Give context to the LLM
     ↓
Generate the answer

```

For conversational RAG:

- Conversation history helps understand what the user means.
- Query rewriting converts that meaning into a retrieval-friendly query.
- Retrieval finds external knowledge.
- The original question remains the user's actual request.
- The LLM uses the retrieved knowledge to generate the response.
- That separation of responsibilities is the core architecture to remember.

#### Key takeaway: RAG

RAG is an architectural pattern for giving an LLM access to external knowledge at inference time.

RAG effectiveness depends on the entire pipeline, not only the LLM.

The most important distinction to remember is:

- Retrieval decides what information the model gets.
- Generation decides how that information becomes an answer.

In conversational RAG:

- History explains the conversation.
- Query rewriting makes the question retrievable.
- Retrieved context provides the knowledge.
- The LLM generates the final answer.

## Lesson 10: Agents

An agent is an LLM-driven decision loop that can choose which available action to take, execute that action through a tool, observe the result, and decide what to do next.

The important distinction not simply that an agent uses tools.

The goal is to understand what makes a system an agent, how agentic execution differs from deterministic  workflow and how Lang chain provides the abstraction needed to build agent

A **deterministic workflow** defines the execution path in advance. An **agent** provides a goal and a bounded set of capabilities, then allows the LLM to decide what action to take next based on the information currently available.

### Deterministic workflow vs. agents

The important distinction is knowing who determines what happens next, not "does the system use tools?".

A deterministic workflow has an execution path defined by the application.  
For example, the RAG system follows known structure:  

```text
User Question
      ↓
Query Processing
      ↓
Retrieval
      ↓
Context Formatting
      ↓
Prompt
      ↓
LLM
      ↓
Answer
```

The application defines this sequence.  

The LLM may perform reasoning or language generation inside individual steps, but it does not decide whether retrieval should happen after the prompt, whether another retrieval operation should happen, or whether the application should suddenly call another capability.  
The work flow is encoded by the application.

An agent is different:

```text
User Goal
    ↓
   LLM
    ↓
Decide Action
    ↓
  Tool
    ↓
Tool Result
    ↓
   LLM
    ↓
Decide Again
   ↙    ↘
Tool   Answer
```

The application still defines the available capabilities and boundaries.  
However the LLM participates in deciding the execution path.

A useful mental model is:  

**Deterministic workflow:** application controls the execution path.   (workflow: Here are the steps you must follow.)  

**Agent:** application controls the available capabilities. LLM controls the decision-making within those boundaries.  
(Agent: here is your goal and here are the capabilities you are allowed to use, decide what to do.)

This does not mean an agent is uncontrolled. The LLM cannot suddenly execute arbitrary python, access an arbitrary database, or send an email unless the application has exposed an appropriate capability.

---

### The agentic loop

An agent can be understood as decision loop around an LLM.


```text
Current Goal + Available Information + Available Tools
              ↓
            LLM
              ↓
        Choose Action
              ↓
      Tool Execution
              ↓
        Tool Result
              ↓
            LLM
              ↓
  Do I have enough information?
        ↙              ↘
      Yes               No
        ↓                 ↓
  Final Answer       Another Action
```

The important distinction is between the mechanics of the loop and the decision inside it.

The runtime knows how to:

1. receive a tool call
2. execute the corresponding tool
3. return the result 
4. provide that result to the model
5. continue or terminate the execution

The dynamic part part is the LLM's choice of what action should happen next.

Therefore, the runtime controls execution mechanics; the LLM controls action selection within the application's boundaries.  
This is the core of agentic execution.

### Why agents exist

Before agent abstraction, an application could manually implement the tool-calling loop:

```text
LLM
 ↓
Inspect response
 ↓
Find requested tool
 ↓
Execute tool
 ↓
Add result to messages
 ↓
Call LLM again
 ↓
Repeat
```

This is possible, but as the system grows, the orchestration becomes more complicated. The application may need to handle:

- multiple tools
- repeated tool calls
- different execution paths
- message state
- termination
- tool errors
- validation
- retries
- observability

An agent abstraction packages this recurring orchestration into a reusable runtime.  
In LangChain:

```python
from langchain.agents import create_agent

agent = create_agent( model=llm, tools=tools, )
```

The abstraction provides the machinery around: LLM → Tool → Result → LLM → ...

### Minimal agent

A minimal agent requires a model and a set of available tools.

Example:

```python

from langchain.agents import create_agent
from langchain_core.tools import tool 

@tool
def get_weather(city: str) -> str:
  """Get the weather for a city.""" 
  return f"The weather in {city} is sunny." 
 
agent = create_agent( 
  model=llm, 
  tools=[get_weather],
  )
```

The important architecture is:

```text
Application
    │
    ├── Model
    │
    └── Tools
          │
          └── get_weather()
```

The model does not directly execute the Python functions.  
The application exposes `get_weather()` as a capability.  
The LLM can request that capability through a structured tool call, and the runtime performs the actual python execution.  

This gives us an important boundary:  
The LLM decides what capability it wants. 
The application decides what that capability actually does.

---

### Tool calling vs agents

Tool calling and agents are closely related, but they are not the same thing.  
Tool calling can be viewed as:  
`LLM → Tool`

The model produces a structured request for a particular tool.

An agent extend this into a decision loop:

```text
LLM
 ↓
Tool
 ↓
Result
 ↓
LLM
 ↓
Tool / Answer
 ↓
...
```

Therefore tool calling is a capability. Agentic execution is a decision loop that can use that capability repeatedly.

Having several tools does not automatically make a system an agent.  
A system could exposes several tools but still use a deterministic workflow: `User → Tool A → Tool B → LLM`.  
If the application always performs the same exact steps, the execution remains deterministic.

The important question is whether the LLM determines the next action.

---

### Reading an agent execution trace

One of the most useful practical skills is being able to read an agent's message sequence.

```text
HumanMessage
    ↓
AIMessage
    └── tool_call: get_weather(...)
    ↓
ToolMessage
    └── weather result
    ↓
AIMessage
    └── final answer
```

The first `AIMessage` represents the model requesting an action.  
The `ToolMessage` represents the result produced by the application after executing that action.  
The final `AIMessage` is generated after the model receives the tool result.

A useful practical distinction is `AIMessage.tool_calls`, when the model wants the runtime to perform an action, the AI message contains tool calls.  
Once the model has enough information and produces the final response, there is no further tool call.  
This distinction becomes especially useful when debugging agents because an agent's output is not necessarily just `final answer`, it can contain the entire execution history.

### Multiple tool calls

An agent does not necessarily have to execute exactly one tool at time.  

Suppose two tools provide independent information:

```text
get_weather(city)
get_temperature(city)
```

THe model may request both in the same response because neither operation depends on the result of the other:

```text
AIMessage
 ├── get_weather("Cairo")
 └── get_temperature("Cairo")
          ↓
    Tool Results
          ↓
      AIMessage
          ↓
     Final Answer
```

The important concept is dependency. If two operations are independent, they maybe requested together.  
If one operation depends on information produced by another, they must be logically sequenced.

for example:

```text
search_product()
      ↓
  product ID
      ↓
get_product_details(product_id)
```

The second operation cannot correctly execute until the first operation has produced the required product ID.  
This leads to one of the most important observations from this lesson:  
>Tool order is often determined by information dependencies, not by the order in which the tools were defined.

### Information Gaps and Tool Dependencies

**Agents close information gaps**  

A useful mental model is to think of an agent as repeatedly closing an information gap. The agent's decision-making process:

```text
What the system currently knows
              ↓
      What is still missing
              ↓
        Possible actions
              ↓
      Choose an action
              ↓
        New information
              ↓
         Re-evaluate
```

Consider: "Tell me the details of a GPU under $600."

Initially, hte agent may know the user's goal but not the internal product ID required by the details tool. It can therefore use:

```python
search_product(
    query="GPU",
    max_price=600
)
```

Suppose the result identifies a product `product ID=2`. Now the agent has information that it did not previously have. It can use `get_product_details(product_id=2)`.  
After receiving that detailed result, it may finally have enough information to answer.  

The important insight is not the particular product example, it is the pattern:
> The result one action can create the information required for the next action.

This is why agentic systems cannot always be represented as one fixed sequence.

**Information dependency determines tool order**  

An important misconception is "If an agent has five tools, it should use all five."  
No. The agent should use the capabilities necessary to satisfy the goal.  

For example suppose the user asks "Do you have GPU?"  
If `search_product("GPU)` already provides enough information to answer the question, there is no reason to call `get_product_details(...)`. The agent can stop.

This gives us another important mental model:

```text
Goal
 ↓
Current information
 ↓
Is the goal satisfied?
 ↙              ↘
Yes              No
 ↓                ↓
Stop          Take another action
```

The agent is therefore not simply executing a predefined list of tools. It is repeatedly evaluating whether another action is necessary.  

---

### Tool Boundaries and Agent Decision-Making

**Tool boundaries**  

A tool is more than a python function decorated with `@tool`.  
From the agent's perspective, a tool is an interface to a capability.

An agent does not directly access the application's backend systems. It reasons over the tool interfaces exposed to it.

The tool's name, description, and schema tell the agent  what capabilities are available and how they can be requested.  
The application remains responsible for executing the selected tool and enforcing validation, authorization, and business rules.

```text
Agent
  ↓
Tool interface
  ↓
Application
  ↓
External system
```

The Tool lesson (lesson 7) covers how to design this interface. In this lesson we focus on how the LLM uses thse interface to decide what action to take next.

**Tool interfaces and agent decision-making**  

The tool lesson established how a tool is exposed to an LLM though its name, description, and input schema. In an agent, this interfaces become the capabilities the model can reason over.  

The important difference is that the agent is not simply following a predefined sequence of tool calls.  
Instead, it considers:

```text
User Goal
    +
Available Capabilities
    +
Information Currently Available
        ↓
Next Action
```

The agent can choose a tool, use the result to update its understanding of the task, and then decide what to do next.  
The tool itself remains an application-controlled capability.  
The agent controls the decision of whether and when to request it.

**Tool interface complexity**  

An agent does not necessarily become better when more tools, or more configurable tools are added.  
The available tools form the agent's action space. If many tools overlap or expose unnecessary complexity, the model has more possibilities to reason over without necessarily gaining useful capability.  

Therefore, the goal is not to maximize the number of tools.  
The goal is to expose the clear set of capabilities that allows the agent to accomplish the intended task.  

This leads to an important agent-design principle:
> Give the agent enough capabilities to accomplish the goal, but avoid unnecessary or overlapping capabilities.


This is different from the detailed tool-design question covered in Lesson 7. Here, hte concern is how the overall set of available capabilities affects the agent's decision-making.

**Information gaps and tool dependencies**  

An agent often needs to acquire information before it can determine its next action.  
The current information available to the agent therefore affects what it can doe next.

For example:

```text
User:
"Tell me the detailed information about a GPU under $600."
        ↓
Information currently available:
No matching product identified
        ↓
search_product(...)
        ↓
Result:
Product ID = 2
        ↓
Information currently available:
A matching product has been identified,
but its details are still unknown
        ↓
get_product_details(2)
        ↓
Detailed result
        ↓
Enough information
        ↓
Final answer
```

The important point is that the agent does not necessarily have a predefined rule such as `search_product → get_product details`.  
Instead, the second action becomes useful because the first action produced information required to continue toward the user's goal.

This create an information dependency:
> If the result of one action provides information required for another action, that dependency can determine the next tool the agent should use.

The can also stop when the information it has is sufficient to accomplish the goal.

An agent is **not** just an LLM that calls tools. it use tools to acquire or transform information needed to accomplish a goal, while dynamically deciding what action is appropriate based on the current state of the task.

---

### RAG vs agents

RAG and agents solve different problems.  

A deterministic RAG workflow answers a questions such as "How do I retrieve relevant external knowledge and provide it to the LLM?".  

An agent answers a different question "What action should the system take next to accomplish user's goal?".

RAG is an information-retrieval architecture.  
An agent is a dynamic decision-making/orchestration architecture.

They can be combined.  
For example, an agent may have a retrieval tool and decide when retrieval is necessary.  
But RAG itself does not imply agentic execution.  
If the retrieval process is already known and stable, adding an agent may simply introduce unnecessary complexity.  

### Production perspective: Agents

In real AI product, an agent would rarely exist as an isolated component.  
A simplified architecture could look like:

```text
      User
        ↓
  Application API
        ↓
    Agent Runtime
        ↓
      LLM
    ↙   ↓   ↘
Search  RAG  Database
  ↓      ↓      ↓
APIs / External Systems
```

The agent provides flexible decision-making.  
The application provides the boundaries.  
The tools provide controlled access to external capabilities.  
The deterministic backend performs the operations that should not depend on probabilistic model behavior.

This division of responsibility is one of the most important architectural ideas to carry into production AI engineering.

**Agents production trade-offs**  

Agents provide flexibility, but flexibility has a cost. Compared with a fixed workflow, agentic execution can introduce:

- additional model calls
- higher latency
- higher cost
- less predictable execution paths
- incorrect tool selection
- invalid tool arguments
- tool failures
- termination problems
- more complicated debugging
- greater observability requirements

This does't mean agents is bad, it means their flexibility should solve an actual engineering problem. A useful production principle is:

> Use an agent when dynamic decision-making provides value, not simply because the system can be made agentic.

**Agents production boundaries**  

A production agent is not simply LLM + several tools. The surrounding application myst still enforce important boundaries.  
A production system may need to control:

- Capabilities.
- Permissions.
- Tool validation.
- Execution limits.
- Timeouts.
- Retries.
- Error handling.
- Termination.
- Observability.
- Cost.
- Latency.

The LLM can decide "I need search for this product."  
But deterministic application code should still decide:

- Whether the request is valid.
- Whether the user has permission.
- Whether the arguments are acceptable.
- Whether the operation is allowed.
- How the backend operation is performed.
- How failures are handled.

This becomes particularly important when tools have side effects, examples include:

- Send an email.
- Modify a database record.
- Create an order.
- Make a purchase.
- Delete data.
- Execute an external operation.

THe LLM should not be treated as a trusted authority over these operations. The applications remains responsible for enforcing boundaries.

### Key Takeaways & Engineering Practices: Agents

**When to use agents?**  

Agents become useful when dynamic-decision-making provides real value.  
Typical condition includes:

- The required action vary depending on hte user's goal.
- The next action depends on information discovered during execution.
- Several capabilities may be relevant.
- The number or order of actions cannot conveniently be hard-coded.
- The system benefits from dynamically selecting capabilities.

Example include:

- Research assistants
- Coding assistants
- Complex customer-support systems
- Systems coordinating multiple external capabilities

**When not to use agents?**  

Don't introduce an agent simply because an agent can solve the problem.  
If the application knows `A → B → C` there may be a little reason to make the LLM decide whether B should happen after A.  

A deterministic workflow is often preferable when:

- the execution path is known
- the sequence is stable
- predictable behavior is important
- latency needs to by tightly controlled
- the task does not require dynamic tool selection

For example the RAG workflow already has a known structure.  
Turning it into an agent merely because retrieval is involved would not automatically improve it.

A useful engineering question is "does the system actually need  the LLM to decide what happens next?". If the answer is no, a deterministic workflow may be sufficient.

#### Common mistakes: Agents

**Mistake 1 — "An agent is just an LLM with tools."**  
  Tools alone do not define an agent. The important characteristic is the dynamic decision loop.  

**Mistake 2 — "An agent must use multiple tools."**  
  An agent can solve a request with a single tool or even decide that no tool is necessary.

**Mistake 3 — "The agent must call every available tool."**  
  The agent should stop once it has enough information to satisfy the goal.

**Mistake 4 — "More tools make an agent more capable."**  
  More capabilities can also create a larger and more difficult decision space.

**Mistake 5 — "The LLM should perform the backend logic."**  
  The LLM should select capabilities and provide structured arguments.  
  Deterministic application code should perform deterministic operations.

**Mistake 6 — "Tool order is arbitrary."**  
  Tool order is often constrained by information dependencies.

**Mistake 7 — "Tool descriptions are just documentation."**  
  Tool descriptions are part of the interface presented to the LLM and can influence tool selection and argument generation.

**Mistake 8 — "The agent has unrestricted application**  
  The agent can only use capabilities exposed through its tool boundary and should remain subject to application-controlled permissions and validation.

**Mistake 9 — "Every RAG system should become an agent."**  
  A predictable RAG workflow does not require agentic execution.

**Mistake 10 — "An agent is inherently better than a deterministic workflow."**  
  The correct architecture depends on the problem.

#### Key mental models: Agents

**Deterministic workflow:**

Application decides the path.  
LLM performs language/reasoning within that path.  

**Agent:**

```text
Application defines capabilities.
        ↓
LLM decides the next action.
        ↓
Runtime executes the action.
        ↓
Result becomes available.
        ↓
LLM decides again.
        ↓
Repeat until the goal is satisfied.
```

**Tool boundary:**

```text
LLM
 ↓
"What capability do I need?"
 ↓
Tool Interface
 ↓
Deterministic Application Code
 ↓
Database / API / System
```

**Information gap:**

```text
Current knowledge
       ↓
What is missing?
       ↓
Choose action
       ↓
New information
       ↓
Re-evaluate
```

**Core engineering principle:**  

```text
LLM
  ↓
Decision-making

Application
  ↓
Deterministic operations

Tools
  ↓
Controlled capability boundary
```

#### Personal observations: Agents

The implementation led to several important conclusions.  

The agent does not magically "know" the next tool. it reasons from the use's goal, the available tools, and the information currently available.

Tool results are not merely outputs. They become new information that can influence subsequent decisions.

Tool order is often determined by information dependencies. In the product example, `get_product_details()`  depends on the product ID produced by `search_product`.

An agent should not blindly execute every available capability. It should stop when the available information is sufficient to satisfy the user goal.

More parameters don't automatically create a better tool interface. Tool design is an API-design problem.

A deterministic RAG pipeline should not become an agent simply because an LLM is involved.

The agent's capabilities are ultimately constrained by the tools and permissions exposed by the application.

The most important distinction from Lesson 9 is:  

- RAG: the application defines the retrieval workflow.  

- Agent: the application defines the capabilities, while the LLM participates in deciding the execution path.