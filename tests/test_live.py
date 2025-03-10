"""
Live tests for OpenAI and Parasail batch processing.
These tests require valid API keys and will make actual API calls.
"""

import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from dotenv import load_dotenv

import pytest

from openai_batch import example_prompts, batch, create_batch_input, run, providers

# Load environment variables from .env file if it exists
env_path = Path(__file__).parents[1] / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Models to use for live tests
LIVE_TEST_MODELS = {
    "openai": "gpt-4",
    "parasail": "meta-llama/Llama-3.1-8B-Instruct",
}

# Maximum number of batch items to test with
MAX_BATCH_ITEMS = 20


# Skip all tests in this module if the required API keys are not available
def check_api_keys():
    openai_key = os.environ.get("OPENAI_API_KEY")
    parasail_key = os.environ.get("PARASAIL_API_KEY")

    missing_keys = []
    if not openai_key:
        missing_keys.append("OPENAI_API_KEY")
    if not parasail_key:
        missing_keys.append("PARASAIL_API_KEY")

    if missing_keys:
        pytest.skip(f"Missing required API keys: {', '.join(missing_keys)}")


# Apply the check to all tests in this module
pytestmark = [
    pytest.mark.live,  # Mark all tests as live tests
    pytest.mark.skipif(
        not (os.environ.get("OPENAI_API_KEY") and os.environ.get("PARASAIL_API_KEY")),
        reason="Missing required API keys for live tests",
    ),
]


@pytest.mark.parametrize(
    "provider",
    providers.all_providers,
    ids=[str(p) for p in providers.all_providers],
)
def test_live_batch_processing_with_main(provider):
    """Test live batch processing with actual API calls using main functions."""
    check_api_keys()

    n = MAX_BATCH_ITEMS

    with TemporaryDirectory() as td:
        prompt_file = Path(td) / "prompts.txt"
        input_file = Path(td) / "batch_input_file.txt"
        output_file = Path(td) / "output.jsonl"
        error_file = Path(td) / "error.jsonl"

        # Create prompts
        example_prompts.main([str(prompt_file), "-n", str(n)])

        # Convert prompts to batch input file
        create_batch_input.main(
            [str(prompt_file), str(input_file), "--model", LIVE_TEST_MODELS[provider.name]]
        )

        # Validate file
        contents = Path(str(input_file)).read_text()
        assert n == len(contents.splitlines())

        api_key = os.environ.get(provider.api_key_env_var)
        assert (
            api_key
        ), f"No API key for {provider.display_name} in env var {provider.api_key_env_var}"

        # Create and submit batch
        batch_id = run.main(
            [
                str(input_file),
                "-p",
                provider.name,
                "--create",
                "--api-key",
                api_key,
                "-o",
                str(output_file),
                "-e",
                str(error_file),
            ]
        )

        assert batch_id, f"Failed to create batch for {provider.display_name}"
        print(f"Created batch {batch_id} for {provider.display_name}")

        # Wait for batch to complete
        print(f"Waiting for batch {batch_id} to complete for {provider.display_name}")
        run.main(
            [
                "-p",
                provider.name,
                "--resume",
                batch_id,
                "--api-key",
                api_key,
                "-o",
                str(output_file),
                "-e",
                str(error_file),
            ]
        )
        print(f"Batch {batch_id} completed for {provider.display_name}")

        # Verify output file exists and has the correct number of entries
        assert output_file.exists(), f"Output file not created for {provider.display_name}"

        # Read output file and verify structure
        output_lines = output_file.read_text().splitlines()
        assert len(output_lines) > 0, f"No output entries for {provider.display_name}"

        # Verify each output entry has the expected structure
        for line in output_lines:
            entry = json.loads(line)["response"]["body"]
            assert "id" in entry, "Missing 'id' in response"
            assert "object" in entry, "Missing 'object' in response"
            assert "created" in entry, "Missing 'created' in response"
            assert "model" in entry, "Missing 'model' in response"
            assert "choices" in entry, "Missing 'choices' in response"
            assert len(entry["choices"]) > 0, "Empty 'choices' in response"

            # Verify choice structure
            choice = entry["choices"][0]
            assert "index" in choice, "Missing 'index' in choice"
            assert "message" in choice, "Missing 'message' in choice"
            assert "role" in choice["message"], "Missing 'role' in message"
            assert "content" in choice["message"], "Missing 'content' in message"


@pytest.mark.parametrize(
    "provider_name",
    ["openai", "parasail"],
    ids=["OpenAI", "Parasail"],
)
def test_live_batch_processing_direct(provider_name):
    """Test live batch processing with actual API calls using Batch object directly."""
    check_api_keys()

    n = MAX_BATCH_ITEMS
    model = LIVE_TEST_MODELS[provider_name]

    with TemporaryDirectory() as td:
        output_file = Path(td) / "output.jsonl"
        error_file = Path(td) / "error.jsonl"
        submission_file = Path(td) / "batch_submission.jsonl"

        # Create a batch with direct API calls
        with batch.Batch(
            submission_input_file=submission_file, output_file=output_file, error_file=error_file
        ) as batch_obj:
            # Add test prompts to the batch
            for i in range(n):
                batch_obj.add_to_batch(
                    model=model,
                    messages=[
                        {"role": "user", "content": f"Write a one-sentence summary of test #{i+1}"}
                    ],
                )

            # The provider is auto-detected from the model
            # Submit, wait for the batch to complete, and download results
            result, output_path, error_path = batch_obj.submit_wait_download()

            # Verify the batch completed successfully
            assert result.status == "completed", f"Batch failed with status: {result.status}"
            assert output_path == str(
                output_file
            ), f"Output path mismatch: {output_path} vs {output_file}"
            assert error_path == str(
                error_file
            ), f"Error path mismatch: {error_path} vs {error_file}"

        # Verify output file exists and has the correct number of entries
        assert output_file.exists(), f"Output file not created for {provider_name}"

        # Read output file and verify structure
        output_lines = output_file.read_text().splitlines()
        assert len(output_lines) == n, f"Expected {n} entries, got {len(output_lines)}"

        # Verify each output entry has the expected structure
        for line in output_lines:
            entry = json.loads(line)["response"]["body"]
            assert "id" in entry, "Missing 'id' in response"
            assert "object" in entry, "Missing 'object' in response"
            assert "created" in entry, "Missing 'created' in response"
            assert "model" in entry, "Missing 'model' in response"
            assert "choices" in entry, "Missing 'choices' in response"
            assert len(entry["choices"]) > 0, "Empty 'choices' in response"

            # Verify choice structure
            choice = entry["choices"][0]
            assert "index" in choice, "Missing 'index' in choice"
            assert "message" in choice, "Missing 'message' in choice"
            assert "role" in choice["message"], "Missing 'role' in message"
            assert "content" in choice["message"], "Missing 'content' in message"
