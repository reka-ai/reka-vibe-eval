"""
$ pip install --upgrade mistral_common
$ pip install --upgrade vllm

# run pixtral large as server
# https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411#server-image
# e.g.:
$ vllm serve mistralai/Pixtral-Large-Instruct-2411 --config-format mistral --load-format mistral --tokenizer_mode mistral --limit_mm_per_prompt 'image=10' --tensor-parallel-size 8

"""

import requests
import json
from typing import Dict, Any
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
from .base_model import BaseVisionModel

class PixtralServer(BaseVisionModel):
    """Pixtral server implementation."""
    
    def __init__(
        self, 
        model_name: str, 
        server_url: str = "127.0.0.1",
        server_port: str = "8000"
    ):
        super().__init__(model_name)
        self.url = f"http://{server_url}:{server_port}/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer token"
        }
        self.system_prompt = self.load_system_prompt()

    def load_system_prompt(self) -> str:
        """Load system prompt from model files."""
        file_path = hf_hub_download(repo_id=self.model_name, filename="SYSTEM_PROMPT.txt")
        with open(file_path, "r") as file:
            system_prompt = file.read()
        today = datetime.today().strftime("%Y-%m-%d")
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        model_name = self.model_name.split("/")[-1]
        return system_prompt.format(
            name=model_name, today=today, yesterday=yesterday
        )

    def generate_response(self, example: Dict[str, Any]) -> str:
        """Generate a response using Pixtral server.
        
        Args:
            example: Dictionary containing media_url and prompt
            
        Returns:
            str: Generated response from Pixtral server
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example["prompt"]},
                    {"type": "image_url", "image_url": {"url": example["media_url"]}},
                ],
            },
        ]
        payload = {"model": self.model_name, "messages": messages}
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(payload)
        )
        return response.json()["choices"][0]["message"]["content"]