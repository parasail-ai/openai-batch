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


def test_batch_submit_and_wait(tmp_path):
    """Test the new submit and wait functionality in Batch class"""
    submission_input_file = tmp_path / "batch.jsonl"
    output_file = tmp_path / "output.jsonl"
    error_file = tmp_path / "error.jsonl"

    # Create a batch with some requests
    batch_obj = batch.Batch(
        submission_input_file=submission_input_file,
        output_file=output_file,
        error_file=error_file,
    )
    batch_obj.add_to_batch(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])

    def mock_server(request: httpx.Request) -> httpx.Response:
        if "files" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "id": "file-abc",
                    "bytes": 100,
                    "created_at": 0,
                    "filename": "test.jsonl",
                    "object": "file",
                    "purpose": "batch",
                    "status": "processed",
                },
            )
        elif "batches" in request.url.path:
            if request.method == "POST":
                return httpx.Response(
                    200,
                    json=openai.types.Batch(
                        id="batch-abc",
                        status="in_progress",
                        completion_window="24h",
                        created_at=0,
                        endpoint="/v1/chat/completions",
                        input_file_id="file-abc",
                        object="batch",
                    ).model_dump(),
                )
            else:  # GET for status check
                return httpx.Response(
                    200,
                    json=openai.types.Batch(
                        id="batch-abc",
                        status="completed",
                        completion_window="24h",
                        created_at=0,
                        endpoint="/v1/chat/completions",
                        input_file_id="file-abc",
                        output_file_id="file-output",
                        error_file_id="file-error",
                        object="batch",
                    ).model_dump(),
                )

    # Mock the OpenAI client
    mock_client = openai.OpenAI(
        http_client=httpx.Client(transport=httpx.MockTransport(mock_server)), api_key="abc"
    )
    batch_obj.provider = batch.get_provider_by_model("gpt-4")
    batch_obj.provider.api_key = "abc"
    batch_obj.provider.base_url = mock_client.base_url

    # Test submit
    batch_id = batch_obj.submit()
    assert batch_id == "batch-abc"
    assert batch_obj.batch_id == "batch-abc"

    # Test wait
    result = batch_obj.wait(interval=0)
    assert result.id == "batch-abc"
    assert result.status == "completed"

    # Test submit_and_wait
    batch_obj = batch.Batch(
        submission_input_file=submission_input_file,
        output_file=output_file,
        error_file=error_file,
    )
    batch_obj.add_to_batch(model="gpt-4", messages=[{"role": "user", "content": "Hello"}])
    batch_obj.provider = batch.get_provider_by_model("gpt-4")
    batch_obj.provider.api_key = "abc"
    batch_obj.provider.base_url = mock_client.base_url

    result = batch_obj.submit_and_wait(interval=0)
    assert result.id == "batch-abc"
    assert result.status == "completed"


@pytest.mark.parametrize(
    "num_iterations, batch_ids",
    [
        (0, "batch-abc"),
        (1, "batch-abc"),
        (0, ["batch-abc"]),
        (0, ["batch-abc", "batch-def", "batch-xyz"]),
        (1, ["batch-abc", "batch-def", "batch-xyz"]),
    ],
    ids=[
        "already done - single batch",
        "in progress - single batch",
        "already done - single batch in list",
        "already done - multiple batches",
        "in progress - multiple batches",
    ],
)
def test_legacy_wait(num_iterations, batch_ids):
    """Test backward compatibility of the wait function"""
    per_batch_counter = {
        bid: num_iterations
        for bid in ([batch_ids] if isinstance(batch_ids, str) else list(batch_ids))
    }

    def mock_server(request: httpx.Request) -> httpx.Response:
        nonlocal per_batch_counter
        request_batch_id = request.url.path.split("/")[-1]
        per_batch_counter[request_batch_id] -= 1

        return httpx.Response(
            200,
            json=openai.types.Batch(
                id=request_batch_id,
                status="completed" if per_batch_counter[request_batch_id] < 0 else "in_progress",
                completion_window="24h",
                created_at=0,
                endpoint="/v1/chat/completions",
                input_file_id="mock-input.jsonl",
                object="batch",
            ).model_dump(),
        )

    mock_client = openai.OpenAI(
        http_client=httpx.Client(transport=httpx.MockTransport(mock_server)), api_key="abc"
    )

    # Test backward compatibility of the wait function
    wait_ret = openai_batch.wait(client=mock_client, batch_id=batch_ids, interval=0)

    # validate expected number of API calls occurred
    for i in per_batch_counter.values():
        assert i == -1

    # validate return value
    if isinstance(batch_ids, str):
        assert isinstance(wait_ret, openai.types.Batch)
        assert wait_ret.id == batch_ids
    else:
        for batch, batch_id in zip(wait_ret, batch_ids):
            assert isinstance(batch, openai.types.Batch)
            assert batch.id == batch_id
