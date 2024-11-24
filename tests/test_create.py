from tempfile import TemporaryDirectory
from pathlib import Path

import pytest

from openai_batch import example_prompts, create_input


@pytest.mark.parametrize(
    "args",
    [
        ["-n", "10"],
        ["-n", "10", "-e"],
    ],
    ids=["chat completion", "embedding"],
)
def test_example_prompts_script(args):
    with TemporaryDirectory() as td:
        prompts = Path(td) / "prompts.txt"
        example_prompts.main([str(prompts)] + args)

        assert 10 == len(prompts.read_text().splitlines())


@pytest.mark.parametrize(
    "embedding",
    [False, True],
    ids=["chat completion", "embedding"],
)
def test_create_input_script(embedding):
    n = 10
    e = ["-e"] if embedding else []

    with TemporaryDirectory() as td:
        prompts = Path(td) / "prompts.txt"
        input_file = Path(td) / "batch_input_file.txt"

        # create prompts
        example_prompts.main([str(prompts), "-n", str(n)] + e)

        # convert prompts to batch input file
        create_input.main([str(prompts), str(input_file)] + e)

        # validate file
        contents = input_file.read_text()
        assert n == len(contents.splitlines())

        for line in contents.splitlines():
            if embedding:
                assert '"input"' in line
            else:
                assert '"messages"' in line
