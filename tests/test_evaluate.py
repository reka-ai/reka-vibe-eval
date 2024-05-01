from evaluate import Example, make_evaluator_prompt


def test__make_evaluator_prompt__no_include_image():
    example = Example(
        example_id="test",
        category="test",
        prompt="User prompt.",
        reference="Reference answer.",
        media_filename="not-used",
        media_url="not-used",
        generation="Model generation",
    )
    assert (
        make_evaluator_prompt(example, include_image=False)
        == """\
[Question]
User prompt.

[Assistant Response]
Model generation

[Ground Truth Response]
Reference answer.

[System]
Rate whether the assistant response correctly matches the ground truth, it's about an image shared by the user.
The rating should be 1-5, where 1 is incorrect and 5 is correct.
Your response should be in the format:
Explanation: (your explanation)
Rating: (int)\
"""
    )


def test__make_evaluator_prompt__include_image():
    example = Example(
        example_id="test",
        category="test",
        prompt="User prompt.",
        reference="Reference answer.",
        media_filename="not-used",
        media_url="not-used",
        generation="Model generation",
    )
    assert (
        make_evaluator_prompt(example, include_image=True)
        == """\
[Question]
User prompt.

[Assistant Response]
Model generation

[Ground Truth Response]
Reference answer.

[System]
Rate whether the assistant response correctly matches the ground truth, in regards to the image above.
The rating should be 1-5, where 1 is incorrect and 5 is correct.
Your response should be in the format:
Explanation: (your explanation)
Rating: (int)\
"""
    )
