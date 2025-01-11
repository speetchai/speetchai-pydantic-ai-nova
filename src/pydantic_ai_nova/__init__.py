"""
Amazon Bedrock Nova Model Integration for pydantic-ai

This package provides integration between Amazon Bedrock's Nova model and pydantic-ai.
"""

from .nova import AmazonNovaModel, AmazonNovaAgentModel, Usage

__version__ = "0.1.0"
__all__ = ["AmazonNovaModel", "AmazonNovaAgentModel", "Usage"] 