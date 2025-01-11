"""
Amazon Bedrock Nova Model Integration for pydantic-ai

This module provides integration between Amazon Bedrock's Nova model and pydantic-ai.
It allows you to use Nova models with the pydantic-ai framework for building AI agents.

Example:
    >>> from pydantic_ai_nova import AmazonNovaModel
    >>> from pydantic_ai import Agent
    >>> 
    >>> # Initialize the model
    >>> model = AmazonNovaModel(model_id='amazon.nova-micro-v1:0', region_name='us-east-1')
    >>> agent = Agent(model=model)
    >>> 
    >>> # Run a simple query
    >>> response = agent.run_sync("How are you?")
    >>> print(response.data)
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, List, Dict, Any, Union
import boto3
import json
from pydantic_ai.models import Model, AgentModel, ModelResponse, ModelSettings
from pydantic_ai.messages import TextPart, ToolCallPart, ModelMessage
from pydantic_ai import Agent
from pydantic_ai.tools import ToolDefinition

@dataclass
class Usage:
    """
    Tracks token usage and other metrics for model interactions.
    
    Attributes:
        request_tokens (int): Number of tokens in the request
        response_tokens (int): Number of tokens in the response
        total_tokens (int): Total number of tokens used
        requests (int): Number of requests made
        successful_requests (int): Number of successful requests
        failed_requests (int): Number of failed requests
        total_time (float): Total time spent in requests
        details (Dict[str, Any]): Additional usage details
    """
    request_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass(init=False)
class AmazonNovaModel(Model):
    """
    Amazon Bedrock Nova model implementation for pydantic-ai.
    
    This class provides integration with Amazon's Nova model through the Bedrock service.
    It implements the pydantic-ai Model interface for seamless integration.
    
    Attributes:
        model_id (str): The Nova model ID to use (e.g., 'amazon.nova-micro-v1:0')
        client (boto3.client): Boto3 client for Bedrock runtime
    """
    model_id: str
    client: boto3.client = field(repr=False)
    region_name: str
    temperature: float
    top_p: float
    system_prompts: list[dict]  # Added system prompts

    def __init__(
        self,
        model_id: str,
        region_name: str = "us-east-1",
        temperature: float = 1,
        top_p: float = 1,
        system_prompts: Optional[list[dict]] = None
    ):
        """
        Initialize the Nova model.
        
        Args:
            model_id (str): The Nova model ID to use
            region_name (str, optional): AWS region name. Defaults to 'us-east-1'
            temperature (float, optional): Temperature for model inference. Defaults to 1
            top_p (float, optional): Top P for model inference. Defaults to 1
            system_prompts (Optional[list[dict]], optional): List of system prompts. Each prompt should be a dict with 'text' key.
        """
        self.model_id = model_id
        self.region_name = region_name
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompts = system_prompts or []
        self.client = boto3.client('bedrock-runtime', region_name=region_name)

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """
        Create an agent model instance for function calling.
        
        Args:
            function_tools (list[ToolDefinition]): List of available function tools
            allow_text_result (bool): Whether to allow text responses
            result_tools (list[ToolDefinition]): List of result processing tools
        
        Returns:
            AgentModel: An instance of AmazonNovaAgentModel
        """
        tools = [self._map_tool_definition(r) for r in function_tools]
        if result_tools:
            tools += [self._map_tool_definition(r) for r in result_tools]

        return AmazonNovaAgentModel(
            client=self.client,
            model_id=self.model_id,
            allow_text_result=allow_text_result,
            tools=tools,
            temperature=self.temperature,
            top_p=self.top_p,
            system_prompts=self.system_prompts
        )

    def name(self) -> str:
        """Get the model name."""
        return f'amazon_nova:{self.model_id}'

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> dict:
        """Map a pydantic-ai tool definition to Nova's format."""
        return {
            "toolSpec": {
                "name": f.name,
                "description": f"**{f.description}**",  # Nova recommends using ** for emphasis
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": f.parameters_json_schema.get("properties", {}),
                        "required": f.parameters_json_schema.get("required", []),
                    }
                }
            }
        }

@dataclass
class AmazonNovaAgentModel(AgentModel):
    """
    Agent model implementation for Nova.
    
    This class handles the actual interactions with the Nova model,
    including request formatting and response processing.
    
    Attributes:
        client (boto3.client): Boto3 client for Bedrock runtime
        model_id (str): The Nova model ID
        allow_text_result (bool): Whether to allow text responses
        tools (list[dict]): List of available tools
        temperature (float): Temperature for model inference
        top_p (float): Top P for model inference
        system_prompts (list[dict]): List of system prompts
    """
    client: boto3.client
    model_id: str
    allow_text_result: bool
    tools: list[dict]
    temperature: float
    top_p: float
    system_prompts: list[dict]  # Added system prompts

    def _prepare_messages(self, messages: list[ModelMessage]) -> list[dict]:
        """Convert pydantic-ai messages to Nova format"""
        nova_messages = []
        
        # Debug print
        print("Input messages:", messages)
        
        for msg in messages:
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'content'):  # Handle UserPromptPart
                        nova_messages.append({
                            "role": "user",
                            "content": [{"text": part.content}]
                        })
                    elif isinstance(part, TextPart):
                        nova_messages.append({
                            "role": "user",
                            "content": [{"text": part.content}]
                        })
                    elif isinstance(part, ToolCallPart):
                        nova_messages.append({
                            "role": "assistant",
                            "content": [{
                                "function_call": {
                                    "name": part.tool_name,
                                    "arguments": json.dumps(part.args)
                                }
                            }]
                        })
        
        # Debug print
        print("Prepared messages:", nova_messages)
        
        return nova_messages

    async def request(
        self, 
        messages: list[ModelMessage], 
        model_settings: Optional[ModelSettings]
    ) -> tuple[ModelResponse, Usage]:
        """Send a request to the Nova model"""
        request_body = {
            "messages": self._prepare_messages(messages),
            "system": self.system_prompts
        }

        if self.tools:
            request_body["toolConfig"] = {
                "tools": self.tools
            }

        try:
            response = await self._invoke_model(request_body)
            return self._process_response(response), Usage()
        except Exception as e:
            print(f"Error in request: {e}")
            raise

    async def _invoke_model(self, request_body: dict) -> dict:
        """Make the actual API call to Nova"""
        try:
            response = self.client.invoke_model(
                body=json.dumps(request_body).encode('utf-8'),
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )
            return json.loads(response['body'].read().decode('utf-8'))
        except Exception as e:
            print(f"Error invoking model: {e}")
            print(f"Request body was: {json.dumps(request_body, indent=2)}")
            raise

    def _process_response(self, response: dict) -> ModelResponse:
        """Process Nova's response into pydantic-ai format"""
        parts = []
        
        # Debug print
        print("Raw response:", response)
        
        if 'output' in response:
            output = response['output']
            if 'message' in output:
                message = output['message']
                if 'content' in message and message['content']:
                    # Handle text response
                    if 'text' in message['content'][0]:
                        parts.append(TextPart(content=message['content'][0]['text']))
                    
                    # Handle function calls
                    if 'tool_calls' in message:
                        for tool_call in message['tool_calls']:
                            parts.append(ToolCallPart.from_raw_args(
                                tool_name=tool_call['function']['name'],
                                args=json.loads(tool_call['function']['arguments']),
                                id=tool_call.get('id', '')
                            ))
                    
                    # Handle single function call
                    if 'function_call' in message:
                        parts.append(ToolCallPart.from_raw_args(
                            tool_name=message['function_call']['name'],
                            args=json.loads(message['function_call']['arguments']),
                            id=message.get('id', '')
                        ))
        
        # Only use default response if we got nothing from the model
        if not parts:
            print("Warning: No response from model, using default response")
            print("Full response was:", response)
            parts.append(TextPart(content="I am an AI assistant. How can I help you?"))
        
        return ModelResponse(parts=parts) 