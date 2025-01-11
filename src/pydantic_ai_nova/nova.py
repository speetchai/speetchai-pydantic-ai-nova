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
from typing import AsyncIterator, Optional, List, Dict, Any
import boto3
import json
from pydantic_ai.models import Model, AgentModel, ModelResponse, ModelSettings
from pydantic_ai.messages import TextPart, ToolCallPart
from pydantic_ai import Agent

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

    def __init__(self, model_id: str, region_name: str = 'us-east-1', **kwargs):
        """
        Initialize the Nova model.
        
        Args:
            model_id (str): The Nova model ID to use
            region_name (str, optional): AWS region name. Defaults to 'us-east-1'
            **kwargs: Additional arguments passed to boto3.client
        """
        self.model_id = model_id
        self.client = boto3.client('bedrock-runtime', region_name=region_name, **kwargs)

    async def agent_model(
        self,
        *,
        function_tools: List,
        allow_text_result: bool,
        result_tools: List,
    ) -> AgentModel:
        """
        Create an agent model instance for function calling.
        
        Args:
            function_tools (List): List of available function tools
            allow_text_result (bool): Whether to allow text responses
            result_tools (List): List of result processing tools
        
        Returns:
            AgentModel: An instance of AmazonNovaAgentModel
        """
        tools = [self._map_tool_definition(tool) for tool in function_tools + result_tools]
        return AmazonNovaAgentModel(
            client=self.client,
            model_id=self.model_id,
            allow_text_result=allow_text_result,
            tools=tools,
        )

    def name(self) -> str:
        """Get the model name."""
        return f'amazon_nova:{self.model_id}'

    @staticmethod
    def _map_tool_definition(tool):
        """Map a pydantic-ai tool definition to Nova's format."""
        return {
            'name': tool.name,
            'description': tool.description,
            'parameters': tool.parameters_json_schema,
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
        tools (List): List of available tools
    """
    client: boto3.client
    model_id: str
    allow_text_result: bool
    tools: List

    async def request(
        self,
        messages: List[Any],
        model_settings: Optional[ModelSettings] = None,
    ) -> (ModelResponse, Usage):
        """
        Send a request to the Nova model.
        
        Args:
            messages (List[Any]): List of messages to send
            model_settings (Optional[ModelSettings]): Model settings
        
        Returns:
            Tuple[ModelResponse, Usage]: Model response and usage statistics
        """
        response = await self._invoke_model(messages, model_settings, stream=False)
        return self._process_response(response), self._extract_usage(response)

    async def request_stream(
        self,
        messages: List[Any],
        model_settings: Optional[ModelSettings] = None,
    ) -> AsyncIterator:
        """
        Send a streaming request to the Nova model.
        
        Args:
            messages (List[Any]): List of messages to send
            model_settings (Optional[ModelSettings]): Model settings
        
        Yields:
            AsyncIterator: Stream of response chunks
        """
        response = await self._invoke_model(messages, model_settings, stream=True)
        async for chunk in self._process_streamed_response(response):
            yield chunk

    def _extract_message_content(self, message: Any) -> str:
        """
        Extract content from various message formats.
        
        Args:
            message (Any): Message to extract content from
        
        Returns:
            str: Extracted message content
        """
        if hasattr(message, 'parts'):
            for part in message.parts:
                if hasattr(part, 'content'):
                    return part.content
        if hasattr(message, 'message'):
            return message.message
        if hasattr(message, 'text'):
            return message.text
        if hasattr(message, 'content'):
            return message.content
        return str(message)

    async def _invoke_model(self, messages: List[Any], model_settings, stream: bool):
        """
        Invoke the Nova model with the given messages.
        
        Args:
            messages (List[Any]): Messages to send
            model_settings: Model settings
            stream (bool): Whether to use streaming mode
        
        Returns:
            Response from the Nova model
        """
        formatted_messages = []
        for msg in messages:
            content = self._extract_message_content(msg)
            if content:
                formatted_messages.append({
                    "role": "user",
                    "content": [{"text": content}]
                })

        # Ensure we have at least one message
        if not formatted_messages:
            formatted_messages.append({
                "role": "user",
                "content": [{"text": "Hello"}]
            })

        payload = {
            "messages": formatted_messages
        }
        
        if model_settings and hasattr(model_settings, 'temperature'):
            payload['temperature'] = model_settings.temperature

        try:
            if stream:
                response = self.client.invoke_model_with_response_stream(
                    body=json.dumps(payload).encode('utf-8'),
                    modelId=self.model_id
                )
                return response
            else:
                response = self.client.invoke_model(
                    body=json.dumps(payload).encode('utf-8'),
                    modelId=self.model_id
                )
                return json.loads(response['body'].read().decode('utf-8'))
        except Exception as e:
            print(f"Error invoking model: {e}")
            print(f"Payload was: {json.dumps(payload, indent=2)}")
            raise

    @staticmethod
    def _process_response(response: Dict[str, Any]) -> ModelResponse:
        """
        Process a response from the Nova model.
        
        Args:
            response (Dict[str, Any]): Raw response from Nova
        
        Returns:
            ModelResponse: Processed response
        """
        parts = []
        
        # Nova returns the response in output.message.content
        if 'output' in response:
            message = response['output'].get('message', {})
            if 'content' in message:
                for content_part in message['content']:
                    if 'text' in content_part:
                        parts.append(TextPart(content=content_part['text']))
        
        # Handle tool calls if present
        if 'tool_calls' in response.get('output', {}).get('message', {}):
            for tool_call in response['output']['message']['tool_calls']:
                tool_name = tool_call.get('function', {}).get('name')
                tool_args = tool_call.get('function', {}).get('arguments', '{}')
                if isinstance(tool_args, str):
                    tool_args = json.loads(tool_args)
                parts.append(ToolCallPart.from_raw_args(
                    tool_name=tool_name,
                    args=tool_args,
                    id=tool_call.get('id', '')
                ))
        
        # Only add default response if we got no response from the model
        if not parts:
            print("Warning: No response from model, using default response")
            print("Response was:", response)
            parts.append(TextPart(content="I am an AI assistant. How can I help you?"))
        
        return ModelResponse(
            parts=parts,
            timestamp=response.get("timestamp"),
        )

    @staticmethod
    async def _process_streamed_response(response):
        """
        Process a streaming response from the Nova model.
        
        Args:
            response: Raw streaming response from Nova
        
        Yields:
            Response parts as they arrive
        """
        async for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'].decode('utf-8'))
            # Handle Nova's output format
            if 'output' in chunk:
                message = chunk['output'].get('message', {})
                if 'content' in message:
                    for content_part in message['content']:
                        if 'text' in content_part:
                            yield TextPart(content=content_part['text'])
                
                # Handle tool calls
                if 'tool_calls' in message:
                    for tool_call in message['tool_calls']:
                        tool_name = tool_call.get('function', {}).get('name')
                        tool_args = tool_call.get('function', {}).get('arguments', '{}')
                        if isinstance(tool_args, str):
                            tool_args = json.loads(tool_args)
                        yield ToolCallPart.from_raw_args(
                            tool_name=tool_name,
                            args=tool_args,
                            id=tool_call.get('id', '')
                        )

    @staticmethod
    def _extract_usage(response: Dict[str, Any]) -> Usage:
        """
        Extract usage statistics from a Nova response.
        
        Args:
            response (Dict[str, Any]): Raw response from Nova
        
        Returns:
            Usage: Usage statistics
        """
        usage_data = response.get("usage", {})
        return Usage(
            request_tokens=usage_data.get("request_tokens", 0),
            response_tokens=usage_data.get("response_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        ) 