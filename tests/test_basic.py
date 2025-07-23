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

    # Test reranker batch
    with open(submission_input_file, "w") as f:
        with batch.Batch(submission_input_file=f) as batch_obj:
            for prompt in prompts:
                batch_obj.add_to_batch(
                    model="rerank-model", text_1=prompt, text_2=f"Reranked {prompt}"
                )

    lines = submission_input_file.read_text().splitlines()
    assert len(lines) == len(prompts)
    for line in lines:
        request = json.loads(line)
        assert request["url"] == "/v1/score"
        assert "text_1" in request["body"]
        assert "text_2" in request["body"]


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


def test_transfusion_batch_validation(tmp_path):
    """Test transfusion batch validation including required params, image conversion, and body content"""
    submission_input_file = tmp_path / "batch.jsonl"

    # Missing required parameters
    with pytest.raises(ValueError, match="Missing required parameters for transfusion requests"):
        with batch.Batch(submission_input_file=submission_input_file) as batch_obj:
            batch_obj.add_to_batch(
                model="Shitao/OmniGen-v1",
                prompt="A beautiful landscape",
                size="1024x1024",
                # Missing image and response_format
            )

    # Missing specific required parameter
    with pytest.raises(
        ValueError,
        match="Missing required parameters for transfusion requests: image",
    ):
        with batch.Batch(submission_input_file=submission_input_file) as batch_obj:
            batch_obj.add_to_batch(
                model="Shitao/OmniGen-v1",
                prompt="A beautiful landscape",
                size="1024x1024",
                response_format="url",
                # Missing image
            )

    # Image as string gets converted to list
    with open(submission_input_file, "w") as f:
        with batch.Batch(submission_input_file=f) as batch_obj:
            batch_obj.add_to_batch(
                model="Shitao/OmniGen-v1",
                prompt="A beautiful landscape",
                size="1024x1024",
                image="https://example.com/image.jpg",
                response_format="url",
            )
    # Verify the request was written correctly
    lines = submission_input_file.read_text().splitlines()
    assert len(lines) == 1
    request = json.loads(lines[0])
    assert request["url"] == "/v1/images/edits"
    assert request["body"]["prompt"] == "A beautiful landscape"
    assert request["body"]["size"] == "1024x1024"
    assert request["body"]["image"] == [
        "https://example.com/image.jpg"
    ]  # Should be converted to list
    assert request["body"]["response_format"] == "url"

    # Image already as list stays as list
    with open(submission_input_file, "w") as f:
        with batch.Batch(submission_input_file=f) as batch_obj:
            batch_obj.add_to_batch(
                model="Shitao/OmniGen-v1",
                prompt="A beautiful landscape",
                size="1024x1024",
                image=[
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.jpg",
                ],
                response_format="url",
            )
    lines = submission_input_file.read_text().splitlines()
    request = json.loads(lines[0])
    assert request["body"]["image"] == [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
    ]

    # Body contains all passed arguments including non-required ones
    with open(submission_input_file, "w") as f:
        with batch.Batch(submission_input_file=f) as batch_obj:
            batch_obj.add_to_batch(
                model="Shitao/OmniGen-v1",
                prompt="A beautiful landscape",
                size="1024x1024",
                image="https://example.com/image.jpg",
                response_format="url",
                # Additional non-required parameters
                quality="hd",
                style="vivid",
                extra_param="extra_value",
            )
    lines = submission_input_file.read_text().splitlines()
    request = json.loads(lines[0])
    body = request["body"]
    # Check required parameters
    assert body["prompt"] == "A beautiful landscape"
    assert body["size"] == "1024x1024"
    assert body["image"] == ["https://example.com/image.jpg"]
    assert body["response_format"] == "url"
    # Check non-required parameters are also included
    assert body["quality"] == "hd"
    assert body["style"] == "vivid"
    assert body["extra_param"] == "extra_value"
    assert body["model"] == "Shitao/OmniGen-v1"

    # Multiple transfusion requests in same batch
    with open(submission_input_file, "w") as f:
        with batch.Batch(submission_input_file=f) as batch_obj:
            # First request
            batch_obj.add_to_batch(
                model="Shitao/OmniGen-v1",
                prompt="First image",
                size="512x512",
                image="https://example.com/image1.jpg",
                response_format="url",
            )
            # Second request
            batch_obj.add_to_batch(
                model="Shitao/OmniGen-v1",
                prompt="Second image",
                size="1024x1024",
                image=["https://example.com/image2.jpg"],
                response_format="b64_json",
            )
    lines = submission_input_file.read_text().splitlines()
    assert len(lines) == 2
    # Check first request
    request1 = json.loads(lines[0])
    assert request1["body"]["prompt"] == "First image"
    assert request1["body"]["image"] == ["https://example.com/image1.jpg"]
    # Check second request
    request2 = json.loads(lines[1])
    assert request2["body"]["prompt"] == "Second image"
    assert request2["body"]["image"] == ["https://example.com/image2.jpg"]
    assert request2["body"]["response_format"] == "b64_json"
