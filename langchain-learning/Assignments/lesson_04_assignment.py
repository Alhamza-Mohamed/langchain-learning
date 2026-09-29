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

# Create the pydantic model
from pydantic import BaseModel, Field

class University (BaseModel):
    """Information about a university"""

    name: str = Field(
        description="The name of the university."
    )  
    country: str = Field(
        description="The name of the country that the university located in."
    )

class Student (BaseModel):
    """Student's information"""
    full_name: str = Field(
        description="The full name of the student."
    )  
    age: int = Field(
        description="The age of the student."
    )  
    major: str = Field(
        description="The major that student specialized in."
    )  
    gpa: float = Field(
        description="The gpa of the student."
    )  
    university: University = Field(
        description="The university that the student is registered in."
    )  

# Create the parser
from langchain_core.output_parsers import PydanticOutputParser
pydantic_parser = PydanticOutputParser(
    pydantic_object= Student
)

print(pydantic_parser.get_format_instructions())

# Build the prompt
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant.
            {format_instructions}
            """
        ),
        ("human", "{question}")
    ]
).partial(format_instructions = pydantic_parser.get_format_instructions() )

# Render the prompt
prompt_value = prompt.invoke(
    {
        "question":"Tell me about a fictional university student."
    }
)

# Invoke the model
response = llm.invoke(prompt_value)

# Invoke the parser
result = pydantic_parser.invoke(response)
print(type(result))
print(result)

print(result.full_name)
print(result.major)
print(result.gpa)
print(result.university.name)
print(result.university.country)

# Convert the result into dict
dict_result = result.model_dump()
print(dict_result)

# Convert the result into JSON
json_result  = result.model_dump_json()
print(json_result )