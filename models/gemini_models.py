"""
$ pip install -q -U google-generativeai
"""

import base64
import google.generativeai as genai
import httpx
import json
import time
from dotenv import load_dotenv
from tqdm import tqdm

MODEL_NAME = "gemini-2.0-flash-exp"
MAX_REQUESTS_PER_MINUTE = 10

load_dotenv()
model = genai.GenerativeModel(model_name=MODEL_NAME)

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
    if media_ext == "jpg":
        media_ext = "jpeg"
    elif media_ext == "png":
        pass
    else:
        print(f"Unexpected image extension for example {example_id}")
        continue
    image = httpx.get(media_url)

    try:
        response = model.generate_content(
            [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image.content).decode("utf-8"),
                },
                prompt,
            ]
        )
        gen = {"example_id": example_id, "generation": response.text}
        generations.append(json.dumps(gen))
        num_requests += 1
    except Exception as e:
        print(f"failed generation for {example_id}, error: {e}")

    if num_requests >= MAX_REQUESTS_PER_MINUTE:
        time.sleep(60)
        num_requests = 0
        start_time = time.time()
with open(f"data/generations/{MODEL_NAME}.jsonl", "w") as fid:
    for gen in generations:
        fid.write(gen)
        fid.write("\n")
