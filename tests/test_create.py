from tempfile import TemporaryDirectory
from pathlib import Path
import io

import pytest

from openai_batch import example_prompts, create_batch_input, create_input, batch


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
    "prompt_args, create_args, expected_line",
    [
        ([], [], "messages"),
        (["-e"], ["-e"], "input"),
        (["-e"], ["--score", "What's good?"], "text_1"),
        (["-e"], ["--rerank", "What's good?"], "documents"),
    ],
    ids=["chat completion", "embedding", "score", "rerank"],
)
def test_create_batch_script(prompt_args, create_args, expected_line):
    n = 10

    with TemporaryDirectory() as td:
        prompts = Path(td) / "prompts.txt"
        input_file = Path(td) / "batch_input_file.txt"

        # create prompts
        example_prompts.main([str(prompts), "-n", str(n)] + prompt_args)

        # convert prompts to batch input file
        create_batch_input.main([str(prompts), str(input_file)] + create_args)

        # validate file
        contents = input_file.read_text()
        assert n == len(contents.splitlines())

        for line in contents.splitlines():
            assert expected_line in line


def test_backwards_compatibility():
    """Test that the old create_input module works the same as create_batch_input"""
    output1 = io.StringIO()
    output2 = io.StringIO()

    # Create batches using both old and new imports
    batch1 = create_input.Batch(output1)
    batch2 = create_batch_input.Batch(output2)

    # Add same requests to both
    for batch in [batch1, batch2]:
        batch.add_to_batch(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])
        batch.add_to_batch(model="gpt-4", messages=[{"role": "user", "content": "World"}])

    # Verify outputs are identical
    assert output1.getvalue() == output2.getvalue()


def test_openai_model_consistency():
    """Test that OpenAI provider enforces model consistency"""
    # Test chat completion batch
    output = io.StringIO()
    batch_instance = batch.Batch(submission_input_file=output)

    # First request with OpenAI model to ensure OpenAI provider
    batch_instance.add_to_batch(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])

    # Same model should work
    batch_instance.add_to_batch(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])

    # Different model should raise error
    with pytest.raises(ValueError, match="Provider openai requires model consistency"):
        batch_instance.add_to_batch(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello"}]
        )

    # Wrong request type should still raise error
    with pytest.raises(ValueError, match="Cannot add embedding to a chat_completion batch"):
        batch_instance.add_to_batch(model="text-embedding-3-large", input="Hello")


def test_parasail_model_mixing():
    """Test that Parasail provider allows mixing models"""
    output = io.StringIO()
    batch_instance = batch.Batch(output)

    # First request with Parasail model
    batch_instance.add_to_batch(
        model="meta-llama/Meta-Llama-3-8B-Instruct", messages=[{"role": "user", "content": "Hello"}]
    )

    # Different model should work with Parasail
    batch_instance.add_to_batch(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
    )


def test_request_type_consistency():
    """Test that request type consistency is always enforced regardless of provider"""
    output = io.StringIO()
    batch_instance = batch.Batch(submission_input_file=output)

    # No input, messages, or text_1
    with pytest.raises(
        ValueError,
        match="Request must include either.*",
    ):
        batch_instance.add_to_batch(model="test", temperature=0.7)

    # Multiple request types
    with pytest.raises(
        ValueError,
        match="Request cannot include multiple types of parameters. Use only one of: .*",
    ):
        batch_instance.add_to_batch(
            model="test", input="Hello", messages=[{"role": "user", "content": "Hello"}]
        )


def test_request_type_consistency_score():
    batch_instance = batch.Batch(submission_input_file=io.StringIO())

    with pytest.raises(ValueError, match="'text_2' is required for score requests"):
        batch_instance.add_to_batch(model="test", text_1="Hello")


def test_request_type_consistency_rerank():
    batch_instance = batch.Batch(submission_input_file=io.StringIO())

    with pytest.raises(ValueError, match="'query' is required for rerank requests"):
        batch_instance.add_to_batch(model="test", documents=["Hello", "World"])

    with pytest.raises(ValueError, match=".*'documents'.*"):
        batch_instance.add_to_batch(model="test", documents=12)

    # auto convert to a list
    batch_instance.add_to_batch(model="test", query="query", documents="Not a list")


def test_batch_type_consistency():
    """Test that batch type consistency is enforced"""

    # Test adding reranker to chat completion batch
    batch_instance = batch.Batch(submission_input_file=io.StringIO())
    batch_instance.add_to_batch(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])
    with pytest.raises(ValueError, match="Cannot add score request to a chat_completion batch"):
        batch_instance.add_to_batch(model="score-model", text_1="Hello", text_2="World")

    # Test adding chat completion to reranker batch
    batch_instance = batch.Batch(submission_input_file=io.StringIO())
    batch_instance.add_to_batch(model="score-model", text_1="Hello", text_2="World")
    with pytest.raises(ValueError, match="Cannot add chat completion to a score batch"):
        batch_instance.add_to_batch(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])

    # Test adding embedding to reranker batch
    batch_instance = batch.Batch(submission_input_file=io.StringIO())
    batch_instance.add_to_batch(model="score-model", text_1="Hello", text_2="World")
    with pytest.raises(ValueError, match="Cannot add embedding to a score batch"):
        batch_instance.add_to_batch(model="text-embedding-3-small", input="Hello")
