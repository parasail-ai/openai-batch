[![Tests](https://github.com/parasail-ai/openai-batch/actions/workflows/tests.yml/badge.svg)](https://github.com/parasail-ai/openai-batch/actions/workflows/tests.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/openai-batch)](https://pypi.org/project/openai-batch/)

# openai-batch

Batch inferencing is an easy and inexpensive way to process thousands or millions of LLM inferences.

The process is:
1. Write inferencing requests to an input file
2. start a batch job
3. wait for it to finish
4. download the output

This library aims to make these steps easier. The OpenAI protocol is relatively easy to use, but it has a lot of boilerplate steps. This library automates those.

#### Supported Providers

* [OpenAI](https://openai.com/) - ChatGPT, GPT4o, etc.
* [Parasail](https://parasail.io/) - Most transformer models on HuggingFace, such as LLama, Qwen, LLava, etc.


## Command-Line Utilities

Use `openai_batch.run` to run a batch from an input file on disk:
```bash
python -m openai_batch.run input.jsonl
```

This will start the batch, wait for it to complete, then download the results.

Useful switches:
* `-c` Only create the batch, do not wait for it.
* `--resume` Attach to an existing batch job. Wait for it to finish then download results.
* `--dry-run` Confirm your configuration without making an actual request.
* Full list: `python -m openai_batch.run --help`

### OpenAI Example
```bash
export OPENAI_API_KEY="<Your OpenAI API Key>"

# Create an example batch input file
python -m openai_batch.example_prompts | \
  python -m openai_batch.create_input --model 'gpt-4o-mini' > input.jsonl

# Run this batch (resumable with `--resume <BATCH_ID>`)
python -m openai_batch.run input.jsonl
```

### Parasail Example

```bash
export PARASAIL_API_KEY="<Your Parasail API Key>"

# Create an example batch input file
python -m openai_batch.example_prompts | \
  python -m openai_batch.create_input --model 'meta-llama/Meta-Llama-3-8B-Instruct' > input.jsonl

# Run this batch (resumable with `--resume <BATCH_ID>`)
python -m openai_batch.run -p parasail input.jsonl
```

## Resources

* [OpenAI Batch Cookbook](https://cookbook.openai.com/examples/batch_processing)
* [OpenAI Batch API reference](https://platform.openai.com/docs/api-reference/batch)
* [OpenAI Files API reference](https://platform.openai.com/docs/api-reference/files)
* [Anthropic's Message Batches](https://www.anthropic.com/news/message-batches-api) - Uses a different API