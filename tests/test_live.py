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
    missing_keys = []

    for provider in providers.all_providers:
        api_key = os.environ.get(provider.api_key_env_var)
        if not api_key:
            missing_keys.append(provider.api_key_env_var)

    if missing_keys:
        pytest.skip(f"Missing required API keys: {', '.join(missing_keys)}")


# Apply the check to all tests in this module
pytestmark = [
    pytest.mark.live,  # Mark all tests as live tests
    pytest.mark.skipif(
        not all(os.environ.get(p.api_key_env_var) for p in providers.all_providers),
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


def test_auto_detect_provider():
    """Test the auto_detect_provider functionality with a real batch job."""
    check_api_keys()

    # Use OpenAI provider for this test
    provider_name = "openai"
    model = LIVE_TEST_MODELS[provider_name]

    with TemporaryDirectory() as td:
        output_file = Path(td) / "output.jsonl"
        error_file = Path(td) / "error.jsonl"
        submission_file = Path(td) / "batch_submission.jsonl"

        # Step 1: Create and submit a batch with a specified provider
        provider_obj = batch.get_provider_by_model(model)
        with batch.Batch(
            submission_input_file=submission_file,
            output_file=output_file,
            error_file=error_file,
            provider=provider_obj,
        ) as batch_obj:
            # Add a single test prompt to the batch
            batch_obj.add_to_batch(
                model=model,
                messages=[{"role": "user", "content": "Write a one-sentence test response."}],
            )

            # Submit the batch
            batch_id = batch_obj.submit()
            print(f"Created batch {batch_id} for auto-detect test")

        # Step 2: Create a new batch object with the batch_id but NO provider
        # This will test the auto_detect_provider functionality
        resumed_batch = batch.Batch(
            batch_id=batch_id,
            output_file=output_file,
            error_file=error_file,
        )

        # Get the status - this should auto-detect the provider
        batch_status = resumed_batch.status()

        # Verify the provider was auto-detected correctly
        assert resumed_batch.provider is not None, "Provider was not auto-detected"
        assert (
            resumed_batch.provider.name == provider_name
        ), f"Expected provider {provider_name}, got {resumed_batch.provider.name}"

        # Wait for the batch to complete
        while batch_status.status not in batch.FINISHED_STATES:
            print(f"Batch status: {batch_status.status}")
            import time

            time.sleep(10)
            batch_status = resumed_batch.status()

        # Download the results - this should also work with the auto-detected provider
        output_path, error_path = resumed_batch.download()

        # Verify the output file exists
        assert Path(output_path).exists(), "Output file was not created"


@pytest.mark.parametrize(
    "provider",
    providers.all_providers,
    ids=[str(p) for p in providers.all_providers],
)
def test_live_batch_processing_direct(provider):
    """Test live batch processing with actual API calls using Batch object directly."""
    check_api_keys()

    n = MAX_BATCH_ITEMS
    model = LIVE_TEST_MODELS[provider.name]

    with TemporaryDirectory() as td:
        output_file = Path(td) / "output.jsonl"
        error_file = Path(td) / "error.jsonl"
        submission_file = Path(td) / "batch_submission.jsonl"

        # Create a batch with direct API calls
        provider_obj = batch.get_provider_by_model(model)
        with batch.Batch(
            submission_input_file=submission_file,
            output_file=output_file,
            error_file=error_file,
            provider=provider_obj,
        ) as batch_obj:
            # Add test prompts to the batch
            for i in range(n):
                batch_obj.add_to_batch(
                    model=model,
                    messages=[
                        {"role": "user", "content": f"Write a one-sentence summary of test #{i+1}"}
                    ],
                )

            # Provider is now passed directly to the Batch constructor
            # Submit, wait for the batch to complete, and download results
            # (submit_wait_download now includes the wait logic)
            result, output_path, error_path = batch_obj.submit_wait_download()

            # Verify the batch completed successfully
            assert result.status == "completed", f"Batch failed with status: {result.status}"
            assert str(output_path) == str(
                output_file
            ), f"Output path mismatch: {output_path} vs {str(output_file)}"

        # Verify output file exists and has the correct number of entries
        assert output_file.exists(), f"Output file not created for {provider.display_name}"

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
