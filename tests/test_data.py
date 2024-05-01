"""Test that the vibe eval dataset is in the right format and check the examples."""

import json
from pathlib import Path

from evaluate import Example

_REPO_DIR = Path(__file__).parents[1]


def _check_example(example: Example):
    assert isinstance(example.example_id, str)
    assert isinstance(example.category, str)
    assert isinstance(example.prompt, str)
    assert isinstance(example.reference, str)
    assert isinstance(example.media_filename, str)
    assert isinstance(example.media_url, str)
    assert example.media_url.startswith("http")


def test__vibe_eval_format():
    jsonl_path = _REPO_DIR / "data" / "vibe-eval.v1.jsonl"
    seen_ids = set()
    with open(jsonl_path) as fh:
        for i, line in enumerate(fh):
            example_dict = json.loads(line)
            example = Example(**example_dict)
            assert (
                example.example_id not in seen_ids
            ), f"Duplicate ID on line {i}: {example.example_id}"
            _check_example(example)
