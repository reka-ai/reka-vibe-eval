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

## Leaderboard 🏆

Vibe-Eval Score (%)

| Model               | all    | hard  | normal|
|---------------------|--------|-------|-------|
| Gemini Flash 2.0    | 67.1   | 52.3  | 75.9  |
| Claude 3.5 Sonnet   | 66.0   | 54.0  | 73.1  |
| GPT-4o              | 64.7   | 52.3  | 72.0  |
| Gemini-1.5 Pro      | 63.8   | 52.3  | 70.6  |
| GPT-4o-mini         | 56.7   | 44.7  | 63.8  |
| Reka Flash          | 56.0   | 39.3† | 65.8  |
| Pixtral Large       | 55.1   | 43.0  | 62.3  |
| Grok Vision Beta    | 54.2   | 37.1  | 64.2  |
| Gemini 1.5 Flash 8b | 54.1   | 44.8  | 59.6  |
| Claude Opus         | 52.8   | 41.8  | 59.2  |
| Pixtral 12b         | 52.5   | 39.3  | 60.4  |
| Claude Haiku        | 48.5   | 31.6  | 58.2  |


† Note we expect the results of Reka models to be worse on the hard-set, as these are, by their very definition, prompts that Core cannot solve.

## Running the evaluation

To run the evaluation, use [evaluate.py](evaluate.py) as follows:

```bash
python evaluate.py generations.jsonl -o out.jsonl
```

(you will have to install a couple of requirements, including the Reka API package with `pip install -r requirements.txt`)

The `generations.jsonl` is expected to contain model generations. It should be a JSONL file where each line is a JSON object with keys `"generation"` and `"example_id"` (matching the dataset).

This will output detailed results to `out.jsonl` and will also print a table of final results to stdout.

## Running the generations

We provide example script to generate responses for Claude, Gemini, OpenAI, Reka, xAI and Pixtral models. Just run e.g. `python models/reka_models.py` make sure you have necessary requirements installed and API keys set, written at the top of each script. These will save the generations to a `.jsonl`. in `data/generations` folder.

Set API keys manually or in a `.env` file:

```bash
REKA_API_KEY=your_api_key
OPENAI_API_KEY=your_api_key
GEMINI_API_KEY=your_api_key
ANTHROPIC_API_KEY=your_api_key
XAI_API_KEY=your_api_key
```

**Note, some image sizes exceeed anthropic's API limit of 5MB, therefore we upload these to chat manually and add them to the generations jsonl

## Citation

```bibtex
@article{padlewski2024vibeeval,
  title={Vibe-Eval: A hard evaluation suite for measuring progress of multimodal language models},
  author={Piotr Padlewski and Max Bain and Matthew Henderson and Zhongkai Zhu and Nishant Relan and Hai Pham and Donovan Ong and Kaloyan Aleksiev and Aitor Ormazabal and Samuel Phua and Ethan Yeo and Eugenie Lamprecht and Qi Liu and Yuqi Wang and Eric Chen and Deyu Fu and Lei Li and Che Zheng and Cyprien de Masson d'Autume and Dani Yogatama and Mikel Artetxe and Yi Tay},
  journal={arXiv preprint arXiv:2405.02287},
  year={2024}
}
```
