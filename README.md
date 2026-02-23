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


## Direct Library Usage

You can also use the library directly in your Python code for more control over the batch processing workflow.

### Basic Usage

```python
import random
from openai_batch import Batch

# Create a batch with random prompts
with Batch() as batch:
    objects = ["cat", "robot", "coffee mug", "spaceship", "banana"]
    for i in range(100):
        batch.add_to_batch(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            temperature=0.7,
            max_completion_tokens=1000,
            messages=[{"role": "user", "content": f"Tell me a joke about a {random.choice(objects)}"}]
        )
    
    # Submit, wait for completion, and download results
    result, output_path, error_path = batch.submit_wait_download()
    print(f"Batch completed with status {result.status} and stored in {output_path}")
```

`batch.add_to_batch` accepts the same format as `chat.completion.create` in OpenAI's Python library, so any chat completion parameters can be included. Parasail supports most transformers on HuggingFace (TODO: link), while OpenAI supports all of their serverless models. 

You can also create embedding batches in a similar way:

```python
with Batch() as batch:
    documents = ["The quick brown fox jumps over the lazy dog", 
                 "Machine learning models can process natural language"]
    
    for doc in documents:
        batch.add_to_batch(
            model="text-embedding-3-small",  # OpenAI embedding model
            input=doc
        )
    
    result, output_path, error_path = batch.submit_wait_download()
```

Or analyze images:
```python
from openai_batch import Batch, data_url

with Batch() as batch:
    images = (p for p in Path("/path/to/images").iterdir() if p.suffix.lower() in {".jpg", ".png", ".webp"})

    for image in images:
        batch.add_to_batch(
            model="Qwen/Qwen3-VL-8B-Instruct",
            max_completion_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url(image)}},
                        {"type": "text", "text": "What is in the image?"},
                    ],
                }
            ],
        )

    result, output_path, error_path = batch.submit_wait_download()
```
See full example script: [image_understanding.py](examples/image_understanding.py)

### Step-by-Step Workflow

For more control, you can break down the process into individual steps:

```python
from openai_batch import Batch
import time

# Create a batch object
batch = Batch(
    submission_input_file="batch_input.jsonl",
    output_file="batch_output.jsonl",
    error_file="batch_errors.jsonl"
)

# Add chat completion requests to the batch
objects = ["cat", "robot", "coffee mug", "spaceship", "banana"]
for i in range(5):
    batch.add_to_batch(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Tell me a joke about a {objects[i]}"}]
    )

# Submit the batch
batch_id = batch.submit()
print(f"Batch submitted with ID: {batch_id}")

# Check status periodically
while True:
    status = batch.status()
    print(f"Batch status: {status.status}")
    
    if status.status in ["completed", "failed", "expired", "cancelled"]:
        break
        
    time.sleep(60)  # Check every minute

# Download results once completed
output_path, error_path = batch.download()
print(f"Output saved to: {output_path}")
print(f"Errors saved to: {error_path}")
```

### Working with Different Providers

The library automatically selects the appropriate provider based on the model:

```python
from openai_batch import Batch

# OpenAI models automatically use the OpenAI provider
openai_batch = Batch()
openai_batch.add_to_batch(
    model="gpt-4o-mini",  # OpenAI model
    messages=[{"role": "user", "content": "Hello, world!"}]
)

# Other models automatically use the Parasail provider
parasail_batch = Batch()
parasail_batch.add_to_batch(
    model="meta-llama/Meta-Llama-3-8B-Instruct",  # Non-OpenAI model
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

You can also explicitly specify a provider:

```python
from openai_batch import Batch
from openai_batch.providers import get_provider_by_name

# Get a specific provider
provider = get_provider_by_name("parasail")

# Create a batch with this provider
batch = Batch(provider=provider)
batch.add_to_batch(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

### Resuming an Existing Batch

```python
from openai_batch import Batch
import time

# Resume an existing batch
batch = Batch(batch_id="batch_abc123")

# Check status in a loop until completed
while True:
    status = batch.status()
    print(f"Batch status: {status.status}")
    
    if status.status == "completed":
        output_path, error_path = batch.download()
        print(f"Output saved to: {output_path}")
        break
    elif status.status in ["failed", "expired", "cancelled"]:
        print(f"Batch ended with status: {status.status}")
        break
        
    time.sleep(60)  # Check every minute
```

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
  python -m openai_batch.create_batch --model 'gpt-4o-mini' > input.jsonl

# Run this batch (resumable with `--resume <BATCH_ID>`)
python -m openai_batch.run input.jsonl
```

### Parasail Example

```bash
export PARASAIL_API_KEY="<Your Parasail API Key>"

# Create an example batch input file
python -m openai_batch.example_prompts | \
  python -m openai_batch.create_batch --model 'meta-llama/Meta-Llama-3-8B-Instruct' > input.jsonl

# Run this batch (resumable with `--resume <BATCH_ID>`)
python -m openai_batch.run -p parasail input.jsonl
```

## Audio Transcription

Batch audio transcription allows you to transcribe thousands of audio files efficiently using OpenAI's Whisper models.

### Basic Usage

The `examples/audio_transcription.py` script provides a complete example for batch audio transcription:

```bash
export OPENAI_API_KEY="your-key-here"

# Transcribe audio files in a directory
python examples/audio_transcription.py --audio-directory ./audio_files

# Use a different Whisper model
python examples/audio_transcription.py --audio-directory ./audio_files --model openai/whisper-tiny

# Test without making actual API calls
python examples/audio_transcription.py --audio-directory ./audio_files --dry-run
```

### Key Features

- **Batch Processing**: Transcribe thousands of audio files efficiently
- **Multiple Formats**: Supports MP3, WAV, M4A, OGG, FLAC, MP4, WEBM
- **Model Selection**: Choose from Whisper models (tiny to large-v3)
- **Progress Tracking**: Real-time progress updates
- **Error Handling**: Graceful handling of failed transcriptions
- **Result Extraction**: Save individual transcripts to text files

### Whisper Models

| Model | Speed | Accuracy | Best Use Case |
|-------|-------|----------|---------------|
| `openai/whisper-tiny` | Fast | Good | Real-time processing, testing |
| `openai/whisper-base` | Fast | Good | Everyday use, moderate accuracy |
| `openai/whisper-small` | Moderate | Very Good | General transcription |
| `openai/whisper-medium` | Slow | Excellent | Professional use |
| `openai/whisper-large-v2` | Slow | Excellent | High accuracy needs |
| `openai/whisper-large-v3` | Slow | Best Available | Production quality |

### Output Files

The script generates several output files:

- **`transcriptions.jsonl`**: Line-delimited JSON with successful transcriptions
- **`transcription_errors.jsonl`**: Line-delimited JSON with failed attempts
- **`transcripts/`**: Individual text files for each successful transcription

### Understanding Batch Processing

Batch processing allows you to submit many transcription requests at once, which is much more efficient than submitting them one at a time.

**Benefits:**
- **Cost Effective**: Batch processing is often cheaper than individual API calls
- **Faster**: All files processed in parallel
- **Scalable**: Handle thousands of files with a single script
- **Resumable**: Check status and resume if needed

**Batch Lifecycle:**
1. **Create**: Build batch configuration with audio files
2. **Submit**: Send batch to OpenAI API
3. **Process**: API processes files in parallel
4. **Complete**: Receive results when ready
5. **Download**: Extract transcription data

### Audio File Format Requirements

- **Base64 Encoding**: Audio files are encoded as base64 data URLs
- **MIME Types**: Supported audio types (audio/wav, audio/m4a, audio/flac, etc.)
- **File Size**: Maximum batch input size varies by provider
- **Parallel Processing**: Each file is processed independently

### Troubleshooting

**No audio files found:**
- Check that your directory path is correct
- Verify audio files have supported extensions (.mp3, .wav, etc.)
- Use absolute paths if needed

**API Key not found:**
- Ensure you've exported the API key: `export OPENAI_API_KEY="your-key"`
- Check with: `echo $OPENAI_API_KEY`

**Batch submission failed:**
- Verify API key is valid
- Check internet connection
- Review API rate limits
- Try smaller batch size

**Processing taking too long:**
- Batch processing is asynchronous
- Script polls for completion status
- Check console output for progress updates
- Larger files take more time

## Resources

* [OpenAI Batch Cookbook](https://cookbook.openai.com/examples/batch_processing)
* [OpenAI Batch API reference](https://platform.openai.com/docs/api-reference/batch)
* [OpenAI Files API reference](https://platform.openai.com/docs/api-reference/files)
* [Whisper Model Documentation](https://platform.openai.com/docs/models/whisper)
