"""Anthropic Claude vision model implementation."""
import base64
import anthropic
from typing import Dict, Any
from .base_model import BaseVisionModel
from .utils import validate_image_url, get_image_data

class ClaudeModel(BaseVisionModel):
    """Claude vision model implementation."""
    
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_response(self, example: Dict[str, Any]) -> str:
        """Generate a response using Claude vision model.
        
        Args:
            example: Dictionary containing media_url and prompt
            
        Returns:
            str: Generated response from Claude
        """
        media_type, _ = validate_image_url(example["media_url"])
        image_data = get_image_data(example["media_url"])
        
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(image_data).decode("utf-8"),
                            },
                        },
                        {"type": "text", "text": example["prompt"]},
                    ],
                }
            ],
        )
        return message.content[0].text