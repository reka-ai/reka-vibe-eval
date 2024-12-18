from typing import Dict, Any
from reka.client import Reka
from reka import ChatMessage
from .base_model import BaseVisionModel

class RekaModel(BaseVisionModel):
    """Reka vision model implementation."""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = Reka(api_key=api_key)

    def generate_response(self, example: Dict[str, Any]) -> str:
        """Generate a response using Reka vision model.
        
        Args:
            example: Dictionary containing media_url and prompt
            
        Returns:
            str: Generated response from Reka
        """
        response = self.client.chat.create(
            messages=[
                ChatMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": example["media_url"],
                        },
                        {
                            "type": "text",
                            "text": example["prompt"],
                        },
                    ],
                    role="user",
                )
            ],
            model=self.model_name,
        )
        return response.responses[0].message.content
