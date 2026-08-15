# Load the environment
from dotenv import load_dotenv
load_dotenv()

# Read the model name
import os
MODEL_NAME = os.environ["GEMINI_MODEL"]
API_KEY = os.environ["GOOGLE_GENERATIVE_AI_API_KEY"]

# Create the LangChain model
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI (model = MODEL_NAME, api_key =API_KEY, temperature = 0)


# Create the LangChain prompt template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages (
    [   
        ("system", """You are a senior computer engineer."""),
        ("human", "{question}")
    ]
)

# StrOutputParser
from langchain_core.output_parsers import StrOutputParser
str_parser = StrOutputParser()

# Create the chain
chain = prompt | llm | str_parser

# Get the invoked result 
invoked_result = chain.invoke(
    {
        "question": "Tell me about async in software"
    }
)

print(type(invoked_result)) # <class 'langchain_core.messages.base.TextAccessor'>
print(invoked_result)


# Get the batched result 
batched_results = chain.batch(
    [
        {"question": "What is LangChain?"},
        {"question": "What is FastAPI?"},
    ]
)

print(type(batched_results)) # List
print(len(batched_results))
for result in batched_results:
    print(result)


# Get the streamed result
for chunk in chain.stream({"question": "Tell me about langGraph"}):
    print(type(chunk)) # <class 'langchain_core.messages.base.TextAccessor'>
    print(chunk)


# Get the ainvoked result
import asyncio

async def result_main():
    result = await chain.ainvoke(
        {
            "question": "Tell me about the most important software principles"
        }
    )
    return(result)

async_result = asyncio.run(result_main())

print (type(async_result)) # <class 'langchain_core.messages.base.TextAccessor'>
print (async_result)

"""
verification

Without changing the chain, answer the following questions.

1- What is the type of the chain itself?
2- Why can chain.invoke() exist even though the chain contains multiple components?
3- What determines the return type of:
    invoke()
    batch()
    stream()
    ainvoke()
4- Explain the difference between:
    invoke()
    batch()
    stream()
    ainvoke()

1- chain is LangChain RunnableSequence which is Runnable itself
2- because all the multiple components are the same type "Runnables" and all of them have the .invoke() function
3- 
    invoke() and ainvoke() returned result's type is determined by the last Runnable, which in our examples is the parser.. 
    batch()'s result is always a List as it consist of multiple questions.
    stream() returns chunks produced by the chain. The type and the behavior of those chunks depend on how the Runnables in the chain support streaming.

4-
    invoke(): 1 sync question return 1 answer as one block
    batch(): multiple sync questions with a list of answers, each answer is for the corresponding question
    stream(): 1 sync question return 1 answer as a stream of chunks, it does not wait for the whole answer to load (unlike .invoke()) but when a chunk is ready it output it
    ainvoke(): 1 async question return 1 answer as one block

For the bonus, it will put a huge load on the free LLM
"""