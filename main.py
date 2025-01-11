from pydantic_ai_nova import AmazonNovaModel
from pydantic_ai import Agent
from typing import List
from pydantic import BaseModel

# Define a function schema
class WeatherParams(BaseModel):
    city: str

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny (this is a mock response)"

# Initialize the model
model = AmazonNovaModel(
    model_id='amazon.nova-micro-v1:0',  # or your preferred Nova model
    region_name='us-east-1'  # your AWS region
)

# Create an agent with functions
agent = Agent(
    model=model,
    function_schemas=[
        {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": WeatherParams.model_json_schema(),
        }
    ],
    functions={
        "get_weather": get_weather
    }
)

# Run a simple query
response = agent.run_sync("How are you?")
print(response.data)

# Use function calling
response = agent.run_sync("What's the weather in Tokyo?")
print(response.data) 