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

products = {
    101: {"name": "Keyboard", "price": 1500, "stock": 12},
    202: {"name": "Mouse", "price": 700, "stock": 25},
}

# Creating tool
from langchain.tools import tool

@tool
def get_product_price(product_id: int):
    """The tool is accept a product_id and return the corresponding product information (the name, price, and the stock) based, it used when the used ask about information about products when provide an id. It should not be used when searching about an information about random items on the internet."""
    product = products[product_id]
    return product

print(get_product_price.name)
print(get_product_price.description)
print(get_product_price.args)

# Direct invoking
result = get_product_price.invoke(
        {
            "product_id": 101
        }
    )

print(result)

# Part 2

# Cerate field enum
from enum import Enum

class ProductField (str, Enum):
    NAME = "name"
    PRICE= "price"
    STOCK= "stock"

field = ProductField("name") # Valid
# field="discount" # ValueError
print(field)



# Create the pydantic
from pydantic import BaseModel

class ProductQuery(BaseModel):
    product_id: int
    field: ProductField

# Part 3

# Create the tool
@tool(args_schema = ProductQuery)
def get_product_price_with_fields(product_id: int, field:ProductField ):
    """The tool is accept a product_id and return the corresponding field (where field is "name", "price", or "stock"), it used when the used ask about information about products when provide an id. It should not be used when searching about an information about random items on the internet."""
    return products[product_id][field] # Could written as products[product_id][field]

print(get_product_price_with_fields.args) # {'product_id': {'title': 'Product Id', 'type': 'integer'}, 'field': {'$ref': '#/$defs/ProductField'}}


# Part 4

# Creates tool object
llm_with_tool = llm.bind_tools([get_product_price_with_fields])

# Invoke the tool
response = llm_with_tool.invoke("What is the price of product 101?")

print(response.tool_calls)

# Part 5

# Create tool call
tool_call = response.tool_calls[0]
print("tool_call: ",tool_call)

tool_args = tool_call['args']
print(tool_args)

# Invoke the tool
result = get_product_price_with_fields.invoke(tool_args) # I know there is an error here and I dont know how to solve it, this is a general problem with the direct invoking.
print ("result: ",result)

# Create the tool message
from langchain_core.messages import ToolMessage

tool_message = ToolMessage(
    content= result,
    tool_call_id = tool_call["id"]
)

# Create the human message
from langchain_core.messages import HumanMessage

messages = [
    HumanMessage (content = "What is the price of product 101"),
    response,
    tool_message,
]

final_response = llm_with_tool.invoke(messages)
print(final_response)

# Part 6

"""
Questions:

What is the difference between a Tool and an ordinary Python function?
What does bind_tools() actually do? Does it execute anything?
What is the difference between a tool_call and Tool execution?
Why does a ToolMessage need the tool_call_id?
Why is a Tool description important to the LLM?
Why shouldn't we rely on the LLM alone to enforce authorization or business rules?
If the LLM has five Tools available, does it have to use one of them?
What's the difference between a deterministic workflow and an Agent?
"""

"""
Answers:

1- A tool is a langchain abstraction that allow an llm to use the function under the abstraction to do specific operation or to get into specific data, the ordinary function cant be run using the LLM

2- No, it just tell the LLM there is a tool available and you can use it when you think you need it.

3- tool_call is done by the LLM to send the args of the tool function to the application, the tool execution is the execution of the args sent by the LLM by the application/python.

4- I think bec it makes the LLM knows what the response (ToolMessage) is about, I mean without it the LLM could not know this message is answer of what request

5- Because it defines the tool for it, so without it (or if it was bad) the LLM wont know clearly about the tool's usage and potential.

6- Because there is specific data that can have limited reach for some persons using the same application, also there is data should not be shared with the LLM itself. This authentication, authorization rules should be separate layer from the LLM.

7- No, it only means the LLM can use them when it thinks they will help to reach its goal 

8- Deterministic workflow is a chain of functions that executed one by one in the same order. Agent is LLM that have the capability of executing different operations based on his decision. 

"""