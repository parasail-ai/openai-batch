# openai-batch

Batch inferencing is an easy and inexpensive way to process thousands or millions of LLM inferences.

The process is:
1. Write inferencing requests to an input file
2. start a batch job
3. wait for it to finish
4. download the output

This library aims to make these steps easier. The OpenAI protocol is relatively easy to use, but it has a lot of boilerplate steps. This library automates those.

## Supported Providers

* [OpenAI](https://openai.com/)
* [Parasail](https://parasail.io/)

## Resources

* [OpenAI Batch Cookbook](https://cookbook.openai.com/examples/batch_processing)
* [OpenAI Batch API reference](https://platform.openai.com/docs/api-reference/batch)
* [OpenAI Files API reference](https://platform.openai.com/docs/api-reference/files)
* [Anthropic's Message Batches](https://www.anthropic.com/news/message-batches-api) - Uses a different API