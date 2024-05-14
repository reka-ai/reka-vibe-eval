# Vibe-Eval

[![main](https://github.com/reka-ai/reka-vibe-eval/actions/workflows/actions.yml/badge.svg)](https://github.com/reka-ai/reka-vibe-eval/actions/workflows/actions.yml)

A benchmark for evaluating multimodal chat models, including especially challenging examples.

[[Link to paper]](https://publications.reka.ai/reka-vibe-eval.pdf) [[Blogpost]](https://www.reka.ai/news/vibe-eval) [[🤗 Dataset]](https://huggingface.co/datasets/RekaAI/VibeEval)

![Example from the dataset](figure.png)

## Dataset

The dataset including all images can be downloaded [in the Releases page of this repo](https://github.com/reka-ai/reka-vibe-eval/releases/tag/v1.0.0).

The dataset is stored as a JSONL file: [data/vibe-eval.v1.jsonl](data/vibe-eval.v1.jsonl).
Each example has the following fields:

- **example_id**: a unique ID for the example
- **category**: the category that this example belongs to, either `difficulty-normal` or `difficulty-hard`
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

## Leaderboard 🏆
Vibe-Eval Score (%)
| Model           | all         | hard       | normal     |
|-----------------|---------------------|--------|--------|
| GPT-4o          | 63.1               | 54.1  | 68.1  |
| Gemini Pro 1.5  | 60.4               | 53.0  | 64.8  |
| GPT-4V          | 57.9               | 46.0  | 64.9  |
| Reka Core       | 53.7               | 38.2† | 62.8  |
| Claude Opus     | 52.8               | 41.8  | 59.2  |
| Reka Flash      | 52.2               | 39.2  | 59.9  |
| Claude Sonnet   | 52.1               | 39.7  | 59.5  |
| Claude Haiku    | 49.8               | 38.5  | 56.4  |
| Llava-1.6-34b   | 48.6               | 39.9  | 53.7  |
| Reka Edge       | 45.4               | 32.2  | 53.1  |
| Llava-1.6-7b    | 43.7               | 35.3  | 48.6  |
| Idefics-2-8b    | 40.0               | 32.2  | 44.6  |
| Idefics-1-80b   | 36.0               | 32.1  | 38.3  |
| Fuyu-8b         | 30.8               | 23.4  | 35.2  |

† Note we expect the results of Reka Core to be worse on the hard-set, as these are, by their very definition, prompts that Core cannot solve.

## Citation

```bibtex
@article{padlewski2024vibeeval,
  title={Vibe-Eval: A hard evaluation suite for measuring progress of multimodal language models},
  author={Piotr Padlewski and Max Bain and Matthew Henderson and Zhongkai Zhu and Nishant Relan and Hai Pham and Donovan Ong and Kaloyan Aleksiev and Aitor Ormazabal and Samuel Phua and Ethan Yeo and Eugenie Lamprecht and Qi Liu and Yuqi Wang and Eric Chen and Deyu Fu and Lei Li and Che Zheng and Cyprien de Masson d'Autume and Dani Yogatama and Mikel Artetxe and Yi Tay},
  journal={arXiv preprint arXiv:2405.02287},
  year={2024}
}
```
