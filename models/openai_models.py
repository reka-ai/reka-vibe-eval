"""
$ pip install openai
"""

from openai import OpenAI
from .base_model import BaseVisionModel
from typing import Dict, Any

class OpenAIModel(BaseVisionModel):
    """OpenAI vision model implementation."""
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, example: Dict[str, Any]) -> str:
        """Generate a response for the given example.
        
        Args:
            example: Dictionary containing at least 'media_url' and 'prompt' keys
            
        Returns:
            str: Generated response from the model
        """
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
        )
        return response.choices[0].message.content