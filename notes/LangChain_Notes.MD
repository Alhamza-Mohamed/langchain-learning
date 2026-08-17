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
      - [Input and Output](#input-and-output)
    - [RunnablePassthrough](#runnablepassthrough)
    - [RunnableParallel](#runnableparallel)
    - [RunnableSequence](#runnablesequence)
      - [RunnableSequence is also a Runnable](#runnablesequence-is-also-a-runnable)
    - [Building a RAG-shaped pipeline](#building-a-rag-shaped-pipeline)
    - [Composition vs Execution](#composition-vs-execution)
    - [Production perspective: Runnables](#production-perspective-runnables)
    - [Key Takeaways \& Engineering Practices: Runnables](#key-takeaways--engineering-practices-runnables)
      - [Key takeaways: Runnables](#key-takeaways-runnables)
      - [Common mistake: Runnables](#common-mistake-runnables)
      - [Best practice: Runnables](#best-practice-runnables)
    - [Personal observations](#personal-observations)

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

#### Input and Output

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

`RunnableParallel` sends the same input through multiple Runnable branches.

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

#### RunnableSequence is also a Runnable

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

### Personal observations

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
