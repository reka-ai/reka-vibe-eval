"""
$ pip install mistral_inference --upgrade
$ pip install --upgrade mistral_common
$ pip install --upgrade vllm
"""

import json
from tqdm import tqdm
from vllm import LLM
from vllm.sampling_params import SamplingParams

MODEL_NAME = "mistralai-pixtral-12b-2409"

sampling_params = SamplingParams(max_tokens=8192, temperature=0.7)

# Lower max_num_seqs or max_model_len on low-VRAM GPUs.
llm = LLM(
    model=MODEL_NAME,
    tokenizer_mode="mistral",
    limit_mm_per_prompt={"image": 5},
    max_model_len=32768,
)

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
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": media_url}},
            ],
        },
    ]

    try:
        outputs = llm.chat(messages=messages, sampling_params=sampling_params)
        gen = {
            "example_id": example_id,
            "generation": outputs[0].outputs[0].text,
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
