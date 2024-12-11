"""
$ pip install openai # we use openai api sdk for simplicity
"""

import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

MODEL_NAME = "grok-vision-beta"
MAX_REQUESTS_PER_HOUR = 60

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("XAI_API_KEY")
client = OpenAI(
    base_url="https://api.x.ai/v1",
)

with open("data/vibe-eval.v1.jsonl", "r") as fid:
    data = fid.readlines()

data = [json.loads(x) for x in data]
generations = []
num_requests = 0
start_time = time.time()
for example in tqdm(data):
    example_id = example["example_id"]
    category = example["category"]
    media_url = example["media_url"]
    prompt = example["prompt"]
    media_ext = media_url.split(".")[-1]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": media_url},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            temperature=0.0,
        )
        gen = {
            "example_id": example_id,
            "generation": response.choices[0].message.content,
        }
        generations.append(json.dumps(gen))
        num_requests += 1
    except Exception as e:
        print(f"failed generation for {example_id}, error: {e}")
    if num_requests >= MAX_REQUESTS_PER_HOUR:
        remaining = 3600 - (time.time() - start_time)
        if remaining > 0:
            time.sleep(remaining)
            num_requests = 0
            start_time = time.time()

with open(f"data/generations/{MODEL_NAME}.jsonl", "w") as fid:
    for gen in generations:
        fid.write(gen)
        fid.write("\n")
