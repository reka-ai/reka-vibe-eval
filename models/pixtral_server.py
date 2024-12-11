"""
$ pip install --upgrade mistral_common
$ pip install --upgrade vllm

# run pixtral large as server
# https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411#server-image
# e.g.:
$ vllm serve mistralai/Pixtral-Large-Instruct-2411 --config-format mistral --load-format mistral --tokenizer_mode mistral --limit_mm_per_prompt 'image=10' --tensor-parallel-size 8

"""

from tqdm import tqdm
import requests
import json
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta

MODEL_NAME = "mistralai/Pixtral-Large-Instruct-2411"
SERVER_URL = "127.0.0.1"  # localhost
SERVER_PORT = "8000"

url = f"http://{SERVER_URL}:{SERVER_PORT}/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(
        name=model_name, today=today, yesterday=yesterday
    )


SYSTEM_PROMPT = load_system_prompt(MODEL_NAME, "SYSTEM_PROMPT.txt")

with open("data/vibe-eval.v1.jsonl", "r") as fid:
    data = fid.readlines()

data = [json.loads(x) for x in data]
generations = []
for example in tqdm(data):
    example_id = example["example_id"]
    category = example["category"]
    media_url = example["media_url"]
    prompt = example["prompt"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": media_url}},
            ],
        },
    ]
    payload = {"model": MODEL_NAME, "messages": messages}
    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload)
        )
        gen = {
            "example_id": example_id,
            "generation": response.json()["choices"][0]["message"]["content"],
        }
        generations.append(json.dumps(gen))
    except Exception as e:
        print(f"failed generation for {example_id}, error: {e}")

with open(
    f"data/generations/{MODEL_NAME.lower().replace('/', '-')}.jsonl", "w"
) as fid:
    for gen in generations:
        fid.write(gen)
        fid.write("\n")
