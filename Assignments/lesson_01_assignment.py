import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the environment
load_dotenv()

# Read the model name
MODEL_NAME = os.environ["GEMINI_MODEL"]
API_KEY = os.environ["GOOGLE_GENERATIVE_AI_API_KEY"]

# Create the LangChain model
llm = ChatGoogleGenerativeAI(model = MODEL_NAME, api_key =API_KEY, temperature = 0)

# System message
from langchain_core.messages import SystemMessage
system_message = SystemMessage(
    content="You are a senior AI researcher"
)

# Human message
from langchain_core.messages import HumanMessage
human_message = HumanMessage(
    content="Explain what is the attention models"
)

# The message
messages = [
    system_message, 
    human_message,
    ]


# Invoke the model
response = llm.invoke(messages)
print(response.content)