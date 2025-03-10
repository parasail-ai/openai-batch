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
    output_file = tmp_path / "batch.jsonl"

    # Test chat completion batch
    with open(output_file, "w") as f:
        with batch.Batch(f) as batch_obj:
            for prompt in prompts:
                batch_obj.add_to_batch(
                    model="gpt-4", messages=[{"role": "user", "content": prompt}]
                )

    lines = output_file.read_text().splitlines()
    assert len(lines) == len(prompts)
    for line in lines:
        request = json.loads(line)
        assert request["url"] == "/v1/chat/completions"
        assert len(request["body"]["messages"]) == 1
        assert request["body"]["messages"][0]["role"] == "user"

    # Test embedding batch
    with open(output_file, "w") as f:
        with batch.Batch(f) as batch_obj:
            for prompt in prompts:
                batch_obj.add_to_batch(model="text-embedding-3-small", input=prompt)

    lines = output_file.read_text().splitlines()
    assert len(lines) == len(prompts)
    for line in lines:
        request = json.loads(line)
        assert request["url"] == "/v1/embeddings"
        assert "input" in request["body"]


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
def test_wait(num_iterations, batch_ids):
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
