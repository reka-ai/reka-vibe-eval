"""
$ pip install mistral_inference --upgrade
$ pip install --upgrade mistral_common
$ pip install --upgrade vllm
"""

from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams
from .base_model import BaseVisionModel
from typing import Dict, Any

class PixtralModel(BaseVisionModel):
    """Pixtral vision model implementation."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)
        self.llm = LLM(
            model=model_name,
            tokenizer_mode="mistral",
            limit_mm_per_prompt={"image": 5},
            max_model_len=32768,
        )

    def generate_response(self, example: Dict[str, Any]) -> str:
        """Generate a response using Pixtral vision model.
        
        Args:
            example: Dictionary containing media_url and prompt
            
        Returns:
            str: Generated response from Pixtral
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": example["prompt"]},
                    {"type": "image_url", "image_url": {"url": example["media_url"]}},
                ],
            },
        ]
        outputs = self.llm.chat(messages=messages, sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text
