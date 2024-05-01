"""Evaluate model generations.

Usage:
    python evaluate.py generations.jsonl -o out.jsonl

The generations.jsonl file is expected to contain generations in the format
{"example_id": "xyz", "generation": "abc"}
(one per line)

This will output a copy of the dataset to out.jsonl with added "generation", "score" and "evaluator_explanation" fields.
"""

import concurrent.futures
import json
import os
import re
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import reka
import requests
import tqdm

_REPO_DIR = Path(__file__).parent


def _parse_args():
    parser = ArgumentParser(description="Vibe-eval evaluation.")
    parser.add_argument(
        "--data",
        type=Path,
        help="Path of the .jsonl file containing the dataset examples.",
        default=_REPO_DIR / "data/vibe-eval.v1.jsonl",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default=Evaluator.REKA_CORE_TEXT.value,
        choices=[e.value for e in Evaluator],
        help="The evaluator to use.",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=8,
        help="Number of concurrent parallel requests to the Reka API to make.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Location to save JSONL file of evaluation results.",
        required=True,
    )
    parser.add_argument(
        "--output_summary",
        type=Path,
        help=(
            "Location to save JSON summarizing evaluation results, if not specified defaults to --output path with "
            "'_summary' suffix added to filename."
        ),
        default=None,
    )
    parser.add_argument(
        "generations",
        type=Path,
        help="JSONL file containing generations, with keys 'example_id' and 'generation'.",
    )
    args = parser.parse_args()
    args.evaluator = Evaluator(args.evaluator)
    return args


class Evaluator(Enum):
    # Use Reka Core (including image input).
    REKA_CORE = "reka-core"

    # Use Reka Core, only using text input.
    REKA_CORE_TEXT = "reka-core-text"


@dataclass
class Example:
    """An example loaded from vibe-eval, stored as jsonl in the repo."""

    example_id: str
    category: str
    prompt: str
    reference: str
    media_filename: str
    media_url: str

    # The fields below are not stored in the dataset, but are populated by this script.
    generation: Optional[str] = None
    score: Optional[int] = None
    evaluator_explanation: Optional[str] = None


_PROMPT_WITH_IMAGE = """\
[Question]
{prompt}

[Assistant Response]
{generation}

[Ground Truth Response]
{reference}

[System]
Rate whether the assistant response correctly matches the ground truth, in regards to the image above.
The rating should be 1-5, where 1 is incorrect and 5 is correct.
Your response should be in the format:
Explanation: (your explanation)
Rating: (int)"""

_PROMPT_WITH_NO_IMAGE = """\
[Question]
{prompt}

[Assistant Response]
{generation}

[Ground Truth Response]
{reference}

[System]
Rate whether the assistant response correctly matches the ground truth, it's about an image shared by the user.
The rating should be 1-5, where 1 is incorrect and 5 is correct.
Your response should be in the format:
Explanation: (your explanation)
Rating: (int)"""


def make_evaluator_prompt(example: Example, include_image: bool) -> str:
    return (_PROMPT_WITH_IMAGE if include_image else _PROMPT_WITH_NO_IMAGE).format(
        prompt=example.prompt,
        reference=example.reference,
        generation=example.generation,
    )


def evaluate(example: Example, evaluator: Evaluator) -> Example:
    """Evaluates the generation and populates the score and explanation fields."""
    include_image = evaluator == Evaluator.REKA_CORE
    evaluator_prompt = make_evaluator_prompt(example, include_image=include_image)
    evaluator_response = reka.chat(
        human=evaluator_prompt,
        media_url=example.media_url if include_image else None,
        temperature=0.4,
        model_name="reka-core-20240415",
        request_output_len=1024,
    )["text"]
    re_match = re.search(r"Rating:\s*([1-5])", evaluator_response)
    if re_match is None:
        raise ValueError(
            f"Evaluator generation did not contain Rating: ([1-5]): {evaluator_response}"
        )
    example.score = int(re_match.group(1))
    example.evaluator_explanation = evaluator_response
    return example


def evaluate_in_parallel_with_retries(
    examples: List[Example],
    evaluator: Evaluator,
    max_retries: int = 10,
    parallelism: int = 8,
    rate_limit_delay: int = 10,  # in seconds
) -> List[Example]:
    """Runs evaluation in parallel, retrying common exceptions."""

    def _evaluate_with_retry(example: Example) -> Example:
        latest_error: BaseException = RuntimeError()
        for i in range(max_retries):
            try:
                return evaluate(example, evaluator=evaluator)
            except Exception as e:
                if (
                    isinstance(e, requests.exceptions.HTTPError)
                    and e.response.status_code == 429
                ):
                    print(
                        f"Hit rate limit error {_exception_debug_str(e)}. Sleeping for {rate_limit_delay}s. Attempt {i + 1} of {max_retries}.",
                        file=sys.stderr,
                    )
                    time.sleep(rate_limit_delay)
                else:
                    print(
                        f"Hit error {_exception_debug_str(e)}. Attempt {i + 1} of {max_retries}.",
                        file=sys.stderr,
                    )
                latest_error = e
        raise latest_error

    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {
            executor.submit(_evaluate_with_retry, example) for example in examples
        }
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(examples)
        ):
            try:
                out.append(future.result())
            except Exception as e:
                for future in futures:
                    future.cancel()
                raise RuntimeError from e

    return out


def _exception_debug_str(e: BaseException) -> str:
    """Generates a string representation of the exception chain."""
    exception_chain = [e]
    while e.__cause__ is not None:
        exception_chain.append(e.__cause__)
        e = e.__cause__
    return " <- ".join(repr(exc) for exc in exception_chain)


def _read_examples(data_fname: Path, generations_fname: Path) -> List[Example]:
    """Create initial Example objects with blank evaluator scores."""
    id_to_example = {}
    id_to_generation = {}

    with data_fname.open() as fh:
        for line in fh:
            example = Example(**json.loads(line))
            id_to_example[example.example_id] = example
    print(
        f"Read {len(id_to_example)} examples from {data_fname}.",
        file=sys.stderr,
    )

    with generations_fname.open() as fh:
        for line in fh:
            obj = json.loads(line)
            id_to_generation[obj["example_id"]] = obj["generation"]
    print(
        f"Read {len(id_to_generation)} examples from {generations_fname}.",
        file=sys.stderr,
    )

    if len(id_to_generation) < len(id_to_example):
        print(f"❗️ Warning: Missing generations for some examples in dataset.")

    examples = []
    for example_id, generation in id_to_generation.items():
        kwargs = asdict(id_to_example[example_id])
        kwargs["generation"] = generation
        examples.append(Example(**kwargs))

    return examples


def _write_examples(examples: List[Example], output_path: Path) -> None:
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as fh:
        for example in examples:
            fh.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")
    print(f"Output {len(examples)} examples to {output_path}.")


def _mean(scores: List[int]) -> float:
    """Scale from 1-5 to 0-100 and compute means."""
    return sum(25 * (score - 1) for score in scores) / len(scores)


def _summarise_metrics(examples: List[Example]) -> None:
    category_to_scores = defaultdict(list)
    for example in examples:
        category_to_scores[example.category].append(example.score)
    print("\n| Category           |  Score       |")
    print("|--------------------|--------------|")

    results = {}
    for category, scores in sorted(category_to_scores.items()):
        score_str = f"{_mean(scores):.2f}"
        print(f"| {category.ljust(18)} | {score_str.ljust(12)} |")
        results[category] = float(score_str)

    overall_score = _mean([example.score for example in examples])
    score_str = f"{overall_score:.2f}"
    print(f"| ALL                | {score_str.ljust(12)} |\n")
    results["overall"] = float(score_str)
    return results


if __name__ == "__main__":
    args = _parse_args()
    if args.output.exists():
        print(
            f"❗️ Warning: --output {args.output} already exists. Will overwrite.",
            file=sys.stderr,
        )
    examples = _read_examples(args.data, args.generations)
    examples = evaluate_in_parallel_with_retries(
        examples=examples,
        evaluator=args.evaluator,
        parallelism=args.parallelism,
    )
    _write_examples(examples, args.output)
    summary = _summarise_metrics(examples)

    # Write summary of metrics to file.
    out_summary = args.output_summary
    if out_summary is None:
        out_base, out_ext = os.path.splitext(args.output)
        out_summary = out_base + "_summary" + out_ext

    with open(out_summary, "w") as fid:
        json.dump(summary, fid)
