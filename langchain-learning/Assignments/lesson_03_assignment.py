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

# Create the LangChain prompt template
from langchain_core.prompts import ChatPromptTemplate
text_prompt = ChatPromptTemplate.from_messages (
    [   
        ("system", """You are a helpful assistant."""),
        ("human", "Tell me about {topic}")
    ]
)

# Render the template
text_prompt_value = text_prompt.invoke (
    {
        "topic": "the movie Interstellar"
    }
)

# Invoke the model
response = llm.invoke(text_prompt_value)

# StrOutputParser
from langchain_core.output_parsers import StrOutputParser
text_parser = StrOutputParser()
text = text_parser.invoke(response)
print (text)


# JsonOutputParser
from langchain_core.output_parsers import JsonOutputParser
json_parser = JsonOutputParser()


# Build the prompt
json_prompt = ChatPromptTemplate.from_messages (
    [   
        (
            "system", 
            """
            You are a helpful assistant.
            {format_instructions}
            """,
        ),
        ("human", "Tell me about {topic}"),
    ]
).partial(format_instructions = json_parser.get_format_instructions())

# Render the template
json_prompt_value = json_prompt.invoke (
    {
        "topic": "the movie Interstellar"
    }
)

# Invoke the model
response = llm.invoke(json_prompt_value)

# Get the result
result = json_parser.invoke(response)
print (type(result))
print (result)
