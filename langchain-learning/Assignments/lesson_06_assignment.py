# Load the environment
from dotenv import load_dotenv
load_dotenv()

# Read the model name
import os
MODEL_NAME = os.environ["GEMINI_MODEL"]
API_KEY = os.environ["GOOGLE_GENERATIVE_AI_API_KEY"]

# Create the LangChain model
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model = MODEL_NAME, api_key = API_KEY, temperature = 0 )

# Part 1

# Create function
def multiply_by_three(x):
    return x * 3

# Convert function into Runnable
from langchain_core.runnables import RunnableLambda
multiply_by_three_runnable = RunnableLambda(multiply_by_three)

# Execute the function

# 1- invoke
invoked_result = multiply_by_three_runnable.invoke(7)
print("invoked_result's type: ",type(invoked_result)) # int
print("invoked_result: ",invoked_result)

# 2- batch
batched_result = multiply_by_three_runnable.batch([1,2,3,4])
print("batched_result's type: ",type(batched_result)) # list
print("batched_result: ",batched_result)

# 3- stream
for chunk in multiply_by_three_runnable.stream(5):
    print("chunk's type: ", type(chunk)) # int
    print("chunk",chunk)

# Part 2

# Fake retriever 
def fake_retriever(question):
    return [" (AI) is computer technology that lets machines think, learn, and solve problems like people. It uses data and patterns instead of strict step-by-step rules " + question] 

retriever = RunnableLambda(fake_retriever)

# Create the chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
parallel_chain = RunnableParallel(
    {
    "question": RunnablePassthrough(),
    "context": retriever 
    }
)

# invoke the chain
parallel_chain_result = parallel_chain.invoke("Tell me about AI")

print("chain's type", type(parallel_chain)) # langchain_core.runnables.base.RunnableParallel
print("chain_result's type", type(parallel_chain_result)) # dict
print("chain_result",parallel_chain_result)

# Part 3

# Create the prompt
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using the provided context"),
        ("human", "Question: {question} \n Context: {context}")
    ]
)

# Create the parser
from langchain_core.output_parsers import StrOutputParser

str_parser = StrOutputParser()

# Create the prompt chain
prompt_chain = parallel_chain | prompt | llm | str_parser

# Invoke the chain
prompt_chain_result = prompt_chain.invoke("Tell me about AI")

print("prompt_chain's type", type(prompt_chain)) # langchain_core.runnables.base.RunnableSequence
print("prompt_chain_result's type", type(prompt_chain_result)) # langchain_core.messages.base.TextAccessor
print("prompt_chain_result",prompt_chain_result)

# Part 4

# Create calculate length function 
from langchain_core.runnables import RunnableLambda

def length_calc (sentence):
    return len(sentence)

length_calc_runnable = RunnableLambda(length_calc)

# Create the chain
length_chain = prompt_chain | length_calc_runnable

# Invoke the chain (I dont think you asked for invoke it but here it is anyway)
length_chain_result = length_chain.invoke("Tell me about AI")

print("length_chain's type", type(length_chain)) # langchain_core.runnables.base.RunnableSequence
print("length_chain_result's type", type(length_chain_result)) # int
print("length_chain_result",length_chain_result)


# Part 5

"""
After the code, answer these in your notebook in your own words:

1- What is a Runnable?

2- Why does LangChain need a common Runnable interface?

3- Why can a RunnableSequence itself be used as a Runnable?

4- What is the difference between RunnableSequence and RunnableParallel?

5- Why does RunnablePassthrough become useful when combined with RunnableParallel?

6- Why do we use RunnableLambda instead of putting an ordinary Python function directly into a Runnable pipeline?

7- What is the difference between:
    chain = prompt | llm | parser

    and:

    chain.invoke(...)
8- What determines whether two Runnables can be meaningfully connected?

Answers:

1- Ok lets be honest here I will write in my words not copy the definition from my notes, Runnable is a abstraction created by LangChain in order to create the same interface for different types of functions/data types so that they share the same methods and can be used in chain.

2- It needs common runnable interface so that they can have the same methods which all can be used with the same way with respecting the input/output type for each runnable, also they can be composed together to form chains using LCEL.

3- It can be used as a Runnable in order to be used in bigger chain itself which let us use the same chain in different bigger chains without re invoking / re writing it

4- RunnableSequence: sequence of Runnable objects where the output of one is the input of the next. 
RunnableParallel: It convert a variable that have different branches (like a dict that have more than 1 key and expects the value) into a chain and sends the same input to those branches
Refinement for answer 5: The dict is not what makes it RunnableParallel, the important thing is that the dict contains Runnable branches, and RunnableParallel executes those branches using the same input and combine their outputs.

5- Bec RunnablePassthrough take the same input and output it without any changes, combining it with RunnableParallel will allow us to give a branches the same input and we can preserve the input without change in one branch by simply using RunnablePassthrough.

6- RunnableLambda allow us to convert a normal function into a Runnable interface which allow us to use it in chain and use the other Runnable methods with it without any problem, putting an ordinary Python function directly into a Runnable pipeline will just be meaning less as you would not be able to use the Runnable methods on the chain, It will be an error.
Refinement for answer 6: In current LangChain LCEL, ordinary callables can be accepted in many composition contexts and automatically coerced into a RunnableLambda, so "it will be an error" is not technically reliable.
The important lesson is: RunnableLambda explicitly wraps/adapts a python callable as Runnable, making the conversion clear and allowing us to treat the function as Runnable object directly.

7- the first is composition as you declare the chain as a sequence of Runnables without any execution, there nothing goes to any runnable in the chain. The second is execution as you tell the chain to start execute the chain with the given input

8- the output of the first is the expected input of the second
Refinement for answer 8: The output of the previous Runnable must be compatible with the input expected by the next Runnable.
"""