from tempfile import NamedTemporaryFile
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
    with NamedTemporaryFile() as prompt_file:
        example_prompts.main([prompt_file.name] + args)
        prompt_file.flush()

        assert 10 == len(Path(prompt_file.name).read_text().splitlines())


@pytest.mark.parametrize(
    "embedding",
    [False, True],
    ids=["chat completion", "embedding"],
)
def test_create_input_script(embedding):
    n = 10
    e = ["-e"] if embedding else []

    with NamedTemporaryFile() as prompt_file, NamedTemporaryFile() as input_file:
        # create prompts
        example_prompts.main([prompt_file.name, "-n", str(n)] + e)
        prompt_file.flush()

        # convert prompts to batch input file
        create_input.main([prompt_file.name, input_file.name] + e)

        # validate file
        contents = Path(input_file.name).read_text()
        assert n == len(contents.splitlines())

        for line in contents.splitlines():
            if embedding:
                assert '"input"' in line
            else:
                assert '"messages"' in line
