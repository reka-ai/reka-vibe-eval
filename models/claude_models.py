"""
$ pip install anthropic
"""

import anthropic
import base64
import httpx
import json
from dotenv import load_dotenv
from tqdm import tqdm


MODEL_NAME = "claude-3-haiku-20240307"

load_dotenv()
client = anthropic.Anthropic()

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
    if media_ext == "jpg":
        media_ext = "jpeg"
    elif media_ext == "png":
        pass
    else:
        print(f"Unexpected image extension for example {example_id}")
        continue
    image_media_type = f"image/{media_ext}"
    image_data = base64.standard_b64encode(
        httpx.get(media_url).content
    ).decode("utf-8")
    try:
        message = client.messages.create(
            model=MODEL_NAME,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        gen = {
            "example_id": example_id,
            "generation": message.content[0].text,
        }
        generations.append(json.dumps(gen))
    except Exception as e:
        print(f"failed generation for {example_id}, error: {e}")

with open(f"data/generations/{MODEL_NAME}.jsonl", "w") as fid:
    for gen in generations:
        fid.write(gen)
        fid.write("\n")
