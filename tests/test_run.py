import json
import os
from tempfile import TemporaryDirectory
from pathlib import Path

import pytest

from openai_batch import example_prompts, batch, create_batch_input, run, providers

CHAT_MODELS = {
    "openai": "gpt-4o-mini",
    "parasail": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def get_dry_run_parameters():
    for p in providers.all_providers:
        yield {"provider": p, "resume": False, "id": f"{str(p)} create"}
        yield {"provider": p, "resume": True, "id": f"{str(p)} resume"}


@pytest.mark.parametrize(
    "provider, resume",
    [(p["provider"], p["resume"]) for p in get_dry_run_parameters()],
    ids=[p["id"] for p in get_dry_run_parameters()],
)
def test_run_script_dry_run(provider, resume):
    with TemporaryDirectory() as td:
        input_file = Path(td) / "batch_input_file.txt"
        with open(input_file, "w") as input:
            with batch.Batch(input) as batch_obj:
                batch_obj.add_to_batch(
                    model="gpt-4", messages=[{"role": "user", "content": "Hello, World!"}]
                )

        if not resume:
            run.main(
                [
                    str(input_file),
                    "-p",
                    provider.name,
                    "--create",
                    "--api-key",
                    "psk-mock",
                    "--dry-run",
                ]
            )
        else:
            run.main(
                [
                    "-p",
                    provider.name,
                    "--resume",
                    "batch-mock",
                    "--api-key",
                    "psk-mock",
                    "--dry-run",
                ]
            )


@pytest.mark.slow
@pytest.mark.parametrize(
    "provider",
    providers.all_providers,
    ids=[str(p) for p in providers.all_providers],
)
def test_run_script_full(provider):
    n = 10

    with TemporaryDirectory() as td:
        prompt_file = Path(td) / "prompts.txt"
        input_file = Path(td) / "batch_input_file.txt"
        output_file = Path(td) / "output.jsonl"
        error_file = Path(td) / "error.jsonl"

        # create prompts
        example_prompts.main([str(prompt_file), "-n", str(n)])

        # convert prompts to batch input file
        create_batch_input.main(
            [str(prompt_file), str(input_file), "--model", CHAT_MODELS[provider.name]]
        )

        # validate file
        contents = Path(str(input_file)).read_text()
        assert n == len(contents.splitlines())

        api_key = os.environ.get(provider.api_key_env_var)

        if not api_key:
            pytest.skip(
                f"No API key for {provider.display_name} in env var {provider.api_key_env_var}"
            )

        # Create batch with dry_run=True
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
                "--dry-run",
            ]
        )
        assert batch_id == "batch-dry-run"

        # wait on batch to complete with dry_run=True
        resumed_batch_id = run.main(
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
                "--dry-run",
            ]
        )
        assert resumed_batch_id == "batch-dry-run"
