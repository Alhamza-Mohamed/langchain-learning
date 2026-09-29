# Part 1

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

from langchain_core.tools import tool

@tool
def search_products(query : str, min_price : int | None=None , max_price: int | None=None) -> list[dict]:
    """
    A tool for search the available products, the arguments are query which is the product name.
    You can set min price and max price.
    You would get the products price and ID if the product is available.
    """
    products = [
        {"name":"GPU", "price":500, "product_id":1},
        {"name":"CPU", "price":300, "product_id":2},
        {"name":"RAM", "price":200, "product_id":4},
    ]
    product = [prod for prod in products 
            if query.lower() in prod["name"].lower() 
            and (max_price is None or prod["price"] <= max_price)
            and (min_price is None or prod["price"] >= min_price)
            ]

    if not product:
        return f"{product} does not exist in the database"
    
    return product
    
    

@tool
def get_product_details (product_id: int)-> dict:
    """
    A tool for returning the details of a product,
    input: the product_id
    output: the details of the product (name, price, manufacture, id)
    """
    products = [
        {"name":"GPU", "price":500, "manufacture":"Nvidia" ,"product_id":1},
        {"name":"CPU", "price":300, "manufacture":"AMD" , "product_id":2},
        {"name":"RAM", "price":200 , "manufacture":"Samsung", "product_id":4},
    ]

    for product_info in products:
        if product_info["product_id"] == product_id:
            return product_info

from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[search_products, get_product_details]
    )

user_query = input("Hello\n")

from langchain_core.output_parsers import StrOutputParser

agent_chain = agent | StrOutputParser()

respond = agent_chain.invoke({
    "messages":[{
        "role":"user",
        "content": user_query
        }]
})

print(respond)

"""
Questions:

1. Why this system benefits from an agent rather than simply hard-coding a workflow.

2. How the agent decides which tool to call.

3. How information from one tool can enable a later tool call.

4. Why product_id is a better input for get_product_details() than asking the LLM to provide the product name again.

5. Why the application, rather than the LLM, remains responsible for actually executing the tool.

6. How the agent knows when to stop.
"""

"""
Answers:

1- Because the user question may vary, he can asks for the availability of a product, may ask its price, or its manufacture.
So this large probabilities benefits from agent rather than thinking of all probabilities and hard coded it in deterministic workflow.

2- I think it decides based on the user query (what does the user wants or asking about), the description of the tools itself 
(the LLM decides if the tool is going to help or not based on the description), and the tool args.
also the result  of previous tool calls

3- May be the arguments of one tool can't be determined from the user query itself, it should determined from the result of other tool.
Like the `search_products` where we got the product id in order to search using it in `get_product_details` in order to get the details of the product

4- Because id is unique while we can have multiple product with the same name.
i.e. Because an ID is a stable, unique identifier for a specific product, whereas a name may not uniquely identify one product.

5- Because LLM is responsible only for using the tool while the application is responsible for running it.
i.e. The LLM decides what action should be taken and with which arguments, while the application actually executes that action and controls what the tool is allowed to do.
The LLM shouldn't directly execute, python, access arbitrary databases, or perform arbitrary side effects.

6- When the answer satisfied the query's question or when it believe that there is no answer exist.
i.e. the LLM evaluates the current information and decides whether another action is necessary or whether it can produce the final answer.
"""