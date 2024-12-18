# models/base_model.py
import json
import time
from tqdm import tqdm
import os
from typing import List, Dict, Any
from abc import ABC, abstractmethod

class BaseVisionModel(ABC):
    """Base class for vision-language models."""
    
    def __init__(self, model_name: str):
        """Initialize the vision model.
        
        Args:
            model_name: Name/identifier of the model
        """
        self.model_name = model_name
        self.output_file_path = f"data/generations/{model_name.lower().replace('/', '-')}.jsonl"
        if os.path.isfile(self.output_file_path):
            # Warning and prompt for user confirmation
            print(f"Warning: The output file '{self.output_file_path}' already exists and may be overwritten.")
            user_input = input("Do you want to continue? (yes/no): ").strip().lower()
            if user_input not in {'yes', 'y'}:
                print("Operation aborted by user.")
                exit(1)
        os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
        self.max_retries = 3

    def load_data(self, data_path: str) -> List[Dict]:
        with open(data_path, "r") as fid:
            data = fid.readlines()
        return [json.loads(x) for x in data]

    @abstractmethod
    def generate_response(self, example: Dict[str, Any]) -> str:
        """Generate a response for the given example.
        
        Args:
            example: Dictionary containing at least 'media_url' and 'prompt' keys
            
        Returns:
            str: Generated response from the model
        """
        pass

    def process_examples(self, data_path: str = "data/vibe-eval.v1.jsonl"):
        data = self.load_data(data_path)
        
        with open(self.output_file_path, "w") as fid:
            for example in tqdm(data):
                example_id = example["example_id"]
                retries = 0
                while retries < self.max_retries:
                    try:
                        response = self.generate_response(example)
                        gen = {
                            "example_id": example_id,
                            "generation": response
                        }
                        fid.write(json.dumps(gen) + "\n")
                        fid.flush()
                        break
                    except Exception as e:
                        retries += 1
                        if retries == 3:
                            print(f"Failed generation for {example_id} after {self.max_retries} retries, error: {e}")
                        else:
                            print(f"Retry {retries}/3 for {example_id}, error: {e}")
                            time.sleep(5)