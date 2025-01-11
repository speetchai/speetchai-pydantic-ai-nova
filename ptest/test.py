from pydantic_ai_nova import AmazonNovaModel
from pydantic_ai import Agent, RunContext
from typing import List
from pydantic import BaseModel
from datetime import datetime

# Define system prompts
SYSTEM_PROMPTS = [
    {"text": "You are an AI assistant that helps users with their requests."},
    {"text": "Model Instructions:"},
    {"text": "- Do not assume any information. All required parameters for actions must come from the User."},
    {"text": "- If you are going to use a tool you should always generate a Thought within <thinking></thinking> tags before you invoke a function."},
    {"text": "- You have access to a weather function that can provide current weather information."},
    {"text": "- When users ask about weather, use the get_weather function instead of suggesting websites."},
    {"text": "- In the Thought, answer:"},
    {"text": "  (1) What is the User's goal?"},
    {"text": "  (2) What information has just been provided?"},
    {"text": "  (3) What is the best action plan?"},
    {"text": "  (4) Are all steps complete? If not, what's next?"},
    {"text": "  (5) Which action is available?"},
    {"text": "  (6) What information does this action require?"},
    {"text": "  (7) Do I have everything I need?"}
]

# Define a function schema
class WeatherParams(BaseModel):
    city: str

def get_weather(ctx: RunContext[WeatherParams]) -> str:
    """
    Mock weather function that returns weather for a city.
    In a real application, this would call a weather API.
    """
    weather_data = {
        "tokyo": {
            "temp": 12,
            "condition": "Partly Cloudy",
            "humidity": 70
        }
    }
    
    city = ctx.params.city.lower()
    if city in weather_data:
        data = weather_data[city]
        return (
            f"Current weather in {city.title()}:\n"
            f"Temperature: {data['temp']}°C\n"
            f"Condition: {data['condition']}\n"
            f"Humidity: {data['humidity']}%\n"
            f"(Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')})"
        )
    return f"Sorry, weather data for {city} is not available."

# Initialize the model with system prompts
model = AmazonNovaModel(
    model_id='amazon.nova-micro-v1:0',
    region_name='us-east-1',
    system_prompts=SYSTEM_PROMPTS
)

# Create an agent
agent = Agent(model=model)

# Register the function
agent._register_function(
    get_weather,
    WeatherParams,
    retries=3,
    prepare=None
)

def run_test(prompt: str):
    print(f"\nTesting prompt: {prompt}")
    response = agent.run_sync(prompt)
    print("Response:", response.data)

# Run tests
#run_test("What is Generative AI?")
run_test("What's the weather in Tokyo?")
#run_test("Is it cold in Tokyo today?") 
