"""
$ pip install openai # xai lets you use openai sdk for simplicity
"""
import os
from typing import Dict, Any
from openai import OpenAI
from .base_model import BaseVisionModel
from .utils import RateLimiter

class XAIModel(BaseVisionModel):
    """X.AI vision model implementation."""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=api_key
        )
        self.rate_limiter = RateLimiter(max_requests=60, time_window=3600)

    def generate_response(self, example: Dict[str, Any]) -> str:
        """Generate a response using X.AI vision model.
        
        Args:
            example: Dictionary containing media_url and prompt
            
        Returns:
            str: Generated response from X.AI
        """
        self.rate_limiter.wait_if_needed()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": example["media_url"]},
                        },
                        {
                            "type": "text",
                            "text": example["prompt"],
                        },
                    ],
                }
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content
