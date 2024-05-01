# Vibe-Eval

[![main](https://github.com/reka-ai/reka-vibe-eval/actions/workflows/actions.yml/badge.svg)](https://github.com/reka-ai/reka-vibe-eval/actions/workflows/actions.yml)

A benchmark for evaluating multimodal chat models, including especially challenging examples.

![Example from the dataset](figure.png)

## Dataset

The dataset including all images can be downloaded [in the Releases page of this repo](https://github.com/reka-ai/reka-vibe-eval/releases/tag/vibe-eval.v1).

The dataset is stored as a JSONL file: [data/vibe-eval.v1.jsonl](data/vibe-eval.v1.jsonl).
Each example has the following fields:

- **example_id**: a unique ID for the example
- **category**: the category that this example belongs to, either `difficulty-normal` or `difficult-hard`
- **prompt**: the user prompt
- **reference**: a golden reference answer for the prompt
- **media_filename**: the name of the file in the dataset
- **media_url**: a URL where the file is hosted publicly

## Running the evaluation

To run the evaluation, use [evaluate.py](evaluate.py) as follows:

```bash
python evaluate.py generations.jsonl -o out.jsonl
```

(you will have to install a couple of requirements, including the Reka API package with `pip install -r requirements.txt`)

The `generations.jsonl` is expected to contain model generations. It should be a JSONL file where each line is a JSON object with keys `"generation"` and `"example_id"` (matching the dataset).

This will output detailed results to `out.jsonl` and will also print a table of final results to stdout.
