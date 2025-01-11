# pydantic-ai-nova

Amazon Bedrock Nova model integration for pydantic-ai.

## Installation

```bash
pip install pydantic-ai-nova
```

## Requirements

- Python 3.8 or higher
- pydantic-ai
- boto3
- AWS credentials configured with access to Amazon Bedrock

## Usage

```python
from pydantic_ai_nova import AmazonNovaModel
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel

# Define system prompts for better function calling
SYSTEM_PROMPTS = [
    {"text": "You are an AI assistant that helps users with their requests."},
    {"text": "Model Instructions:"},
    {"text": "- Do not assume any information. All required parameters for actions must come from the User."},
    {"text": "- If you are going to use a tool you should always generate a Thought within <thinking></thinking> tags before you invoke a function."},
    {"text": "- You have access to a weather function that can provide current weather information."},
    {"text": "- When users ask about weather, use the get_weather function instead of suggesting websites."}
]

# Define your function parameters
class WeatherParams(BaseModel):
    city: str

def get_weather(ctx: RunContext[WeatherParams]) -> str:
    """Get current weather for a city"""
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
            f"Humidity: {data['humidity']}%"
        )
    return f"Sorry, weather data for {city} is not available."

# Initialize the model
model = AmazonNovaModel(
    model_id='amazon.nova-micro-v1:0',  # or your preferred Nova model
    region_name='us-east-1',  # your AWS region
    system_prompts=SYSTEM_PROMPTS  # customize model behavior
)

# Create an agent
agent = Agent(model=model)

# Register your function
agent._register_function(
    get_weather,
    WeatherParams,
    retries=3,
    prepare=None
)

# Run a simple query
response = agent.run_sync("How are you?")
print(response.data)

# Use function calling
response = agent.run_sync("What's the weather in Tokyo?")
print(response.data)

# Ask follow-up questions
response = agent.run_sync("Is it cold there?")
print(response.data)
```

## Features

- Full integration with pydantic-ai's Agent framework
- Support for both streaming and non-streaming responses
- Function calling support
- Token usage tracking
- AWS authentication handling through boto3

## Configuration

The package uses boto3 for AWS authentication. Make sure you have your AWS credentials configured either through:
- Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
- AWS credentials file (`~/.aws/credentials`)
- IAM role when running on AWS services

## Development

To contribute to this project:

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests:
   ```bash
   pytest
   ```

## License

MIT License - see LICENSE file for details. 