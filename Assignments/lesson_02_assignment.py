# Load the environment
from dotenv import load_dotenv
load_dotenv()

# Read the model name
import os
MODEL_NAME = os.environ["GEMINI_MODEL"]
API_KEY = os.environ["GOOGLE_GENERATIVE_AI_API_KEY"]

# Create the LangChain model
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model = MODEL_NAME, api_key = API_KEY, temperature = 0)



# Create LangChain prompt template
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a senior {job} instructor."),
        ("human", "Explain {topic} for a {level}")
    ]
)

# Render the template
prompt_value = prompt.invoke(
    {
        "job": "Ai engineer",
        "topic": "LangChain",
        "level": "intermediate",

    }
)

# Invoke the model 
response = llm.invoke(prompt_value) # response is an AIMessage

# Build  history 
from langchain_core.messages import (HumanMessage, AIMessage)
history = [
    HumanMessage(
        content="Explain LangChain for an intermediate."
    ),
    response,
]


# Create second prompt
from langchain_core.prompts import MessagesPlaceholder
prompt2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
# Render
prompt_value2 = prompt2.invoke(
    {
        "history": history,
        "question": "can you explain it more simply?",
    }
)

# Call LLM 
response2 = llm.invoke(prompt_value2)

print(response2)
