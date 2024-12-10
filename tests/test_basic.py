import httpx
import openai
import pytest

import openai_batch


def test_version():
    assert openai_batch.__version__


def test_batch_create_array():
    prompts = ["Say Pong", "Hello"]
    pass


@pytest.mark.parametrize(
    "num_iterations",
    [0, 1],
    ids=["already done", "needs loop"],
)
def test_wait(num_iterations):
    batch_id = "batch-abcdef"

    def mock_server(request: httpx.Request) -> httpx.Response:
        nonlocal num_iterations
        num_iterations -= 1

        return httpx.Response(
            200,
            json=openai.types.Batch(
                id=request.url.path.split("/")[-1],
                status="completed" if num_iterations < 0 else "in_progress",
                completion_window="24h",
                created_at=0,
                endpoint="/v1/chat/completions",
                input_file_id="mock-input.jsonl",
                object="batch",
            ).model_dump(),
        )

    batch = openai_batch.wait(
        client=openai.OpenAI(
            http_client=httpx.Client(transport=httpx.MockTransport(mock_server)),
            api_key="abc",
        ),
        batch_id=batch_id,
        interval=0,
    )

    assert isinstance(batch, openai.types.Batch)
    assert batch.id == batch_id
