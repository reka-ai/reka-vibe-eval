"""
$ pip install -q -U google-generativeai
"""

import base64
import google.generativeai as genai
from typing import Dict, Any
from .base_model import BaseVisionModel
from .utils import RateLimiter, validate_image_url, get_image_data

class GeminiModel(BaseVisionModel):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name=model_name)
        self.rate_limiter = RateLimiter(max_requests=10, time_window=60)

    def generate_response(self, example: Dict[str, Any]) -> str:
        self.rate_limiter.wait_if_needed()
        
        media_type, _ = validate_image_url(example["media_url"])
        image_data = get_image_data(example["media_url"])
        
        response = self.model.generate_content(
            [
                {
                    "mime_type": media_type,
                    "data": base64.b64encode(image_data).decode("utf-8"),
                },
                example["prompt"],
            ]
        )
        return response.text