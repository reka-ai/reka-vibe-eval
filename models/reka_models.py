import json
import os
from dotenv import load_dotenv
from reka.client import Reka
from reka import ChatMessage
from tqdm import tqdm

MODEL_NAME = "reka-flash"

load_dotenv()
client = Reka()

with open("data/vibe-eval.v1.jsonl", "r") as fid:
    data = fid.readlines()

data = [json.loads(x) for x in data]
generations = []
for example in tqdm(data):
    example_id = example["example_id"]
    category = example["category"]
    media_url = example["media_url"]
    prompt = example["prompt"]
    media_ext = media_url.split(".")[-1]

    try:
        response = client.chat.create(
            messages=[
                ChatMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": media_url,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                    role="user",
                )
            ],
            model=MODEL_NAME,
        )
        gen = {
            "example_id": example_id,
            "generation": response.responses[0].message.content,
        }
        generations.append(json.dumps(gen))
    except Exception as e:
        print(f"failed generation for {example_id}, error: {e}")

with open(f"data/generations/{MODEL_NAME}.jsonl", "w") as fid:
    for gen in generations:
        fid.write(gen)
        fid.write("\n")
