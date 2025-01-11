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
from pydantic_ai import Agent

# Initialize the model
model = AmazonNovaModel(
    model_id='amazon.nova-micro-v1:0',  # or your preferred Nova model
    region_name='us-east-1'  # your AWS region
)

# Create an agent
agent = Agent(model=model)

# Run a simple query
response = agent.run_sync("How are you?")
print(response.data)

# Use function calling
tools = [
    # your function tools here
]

response = agent.run_sync(
    "What's the weather in Tokyo?",
    function_tools=tools
)
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