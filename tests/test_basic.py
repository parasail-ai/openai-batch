import httpx
import json
import openai
import pytest

import openai_batch
from openai_batch import batch


def test_version():
    assert openai_batch.__version__


def test_batch_create_array(tmp_path):
    prompts = ["Say Pong", "Hello"]
    submission_input_file = tmp_path / "batch.jsonl"

    # Test chat completion batch
    with open(submission_input_file, "w") as f:
        with batch.Batch(submission_input_file=f) as batch_obj:
            for prompt in prompts:
                batch_obj.add_to_batch(
                    model="gpt-4", messages=[{"role": "user", "content": prompt}]
                )

    lines = submission_input_file.read_text().splitlines()
    assert len(lines) == len(prompts)
    for line in lines:
        request = json.loads(line)
        assert request["url"] == "/v1/chat/completions"
        assert len(request["body"]["messages"]) == 1
        assert request["body"]["messages"][0]["role"] == "user"

    # Test embedding batch
    with open(submission_input_file, "w") as f:
        with batch.Batch(submission_input_file=f) as batch_obj:
            for prompt in prompts:
                batch_obj.add_to_batch(model="text-embedding-3-small", input=prompt)

    lines = submission_input_file.read_text().splitlines()
    assert len(lines) == len(prompts)
    for line in lines:
        request = json.loads(line)
        assert request["url"] == "/v1/embeddings"
        assert "input" in request["body"]


def test_batch_operations(tmp_path):
    """Test the submit, wait, and download functionality in Batch class using dry_run mode"""
    submission_input_file = tmp_path / "batch.jsonl"
    output_file = tmp_path / "output.jsonl"
    error_file = tmp_path / "error.jsonl"

    # Create a batch with some requests
    provider = batch.get_provider_by_model("gpt-4")
    batch_obj = batch.Batch(
        submission_input_file=submission_input_file,
        output_file=output_file,
        error_file=error_file,
        provider=provider,
    )
    batch_obj.add_to_batch(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])

    # Test submit with dry_run=True
    batch_id = batch_obj.submit(dry_run=True)
    assert batch_id == "batch-dry-run"
    assert batch_obj.batch_id == "batch-dry-run"

    # Test status with dry_run=True
    result = batch_obj.status(dry_run=True)
    assert result.id == "batch-dry-run"
    assert result.status == "completed"

    # Test download with dry_run=True
    output_path, error_path = batch_obj.download(dry_run=True)
    assert str(output_path) == str(output_file)
    assert str(error_path) == str(error_file)
    assert output_file.exists()
    assert error_file.exists()

    # Test submit_wait_download with dry_run=True
    provider = batch.get_provider_by_model("gpt-4")
    batch_obj = batch.Batch(
        submission_input_file=submission_input_file,
        output_file=output_file,
        error_file=error_file,
        provider=provider,
    )
    batch_obj.add_to_batch(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])

    result, output_path, error_path = batch_obj.submit_wait_download(interval=0, dry_run=True)
    assert result.id == "batch-dry-run"
    assert result.status == "completed"
    assert str(output_path) == str(output_file)
    assert str(error_path) == str(error_file)


def test_legacy_wait():
    """Test backward compatibility of the wait function"""
    # Note: This test would need to be updated in the actual openai_batch module
    # to support dry_run mode. For now, we're just testing that the function exists.
    assert hasattr(openai_batch, "wait")


def test_batch_validation():
    """Test validation rules for Batch creation and usage"""
    # Test that providing both submission_input_file and batch_id raises an error
    with pytest.raises(ValueError, match="Cannot specify both submission_input_file and batch_id"):
        batch.Batch(submission_input_file="input.jsonl", batch_id="batch-123")

    # Test that adding to a batch with batch_id set raises an error
    provider = batch.get_provider_by_model("gpt-4")
    batch_obj = batch.Batch(batch_id="batch-123", provider=provider)
    with pytest.raises(ValueError, match="Adding to an existing batch is not supported"):
        batch_obj.add_to_batch(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])
