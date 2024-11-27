import openai_batch


def test_version():
    assert openai_batch.__version__


def test_batch_create_array():
    prompts = ["Say Pong", "Hello"]
    pass
