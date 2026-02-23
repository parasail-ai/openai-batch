# AGENTS.md

This file contains conventions and commands for agentic coding assistants working in this repository.

## Build, Lint, and Test Commands

### Format Code
```bash
# Format all code in the repository
python -m black openai_batch/ tests/ examples/

# Format only specific files
python -m black openai_batch/batch.py openai_batch/providers.py
```

### Run Tests
```bash
# Run all tests (excludes slow and live tests by default)
pytest

# Run a single test
pytest tests/test_basic.py::test_version

# Run tests in a specific file
pytest tests/test_basic.py

# Run slow tests (may make API calls)
pytest -m slow

# Run live tests (requires actual API keys)
pytest -m live

# Run all tests including slow and live
pytest -m 'slow or live'

# Run with coverage (if coverage is installed)
pytest --cov=openai_batch --cov-report=html
```

## Code Style Guidelines

### Formatting
- Use Black formatter with line length 100
- Target Python 3.8+
- Apply Black formatting to all new files

### Imports
- Organize imports: stdlib → third-party → local (separated by blank lines)
- Import specific classes/functions rather than modules
- Use `from typing import ...` for type hints
- Import OpenAI types: `from openai.types import *` for batch/embedding/chat types
- Import custom modules: `from openai_batch import Batch, data_url`

### Type Hints
- Always include type hints for function signatures
- Use `Optional[T]` for nullable parameters
- Use `Union` for multiple possible types
- Use `Tuple[...]` for multiple return values
- Import from `typing` module
- Use `OpenAIBatch` type from `openai.types.batch`

### Naming Conventions
- **Classes**: PascalCase (Batch, Provider, BatchType)
- **Functions/variables**: snake_case (add_to_batch, submission_input_file)
- **Constants**: CAPS_SNAKE_CASE (FINISHED_STATES)
- **Private methods**: prefix with underscore (_ensure_submission_file)
- **Enums**: PascalCase (BatchType) with lowercase values

### File and Path Handling
- Use `pathlib.Path` instead of string paths for file operations
- Use encoding="utf-8" when opening files
- Use context managers (`with` statements) for file I/O
- Use `Path.resolve()` for absolute paths
- Use `Path.mkdir(parents=True, exist_ok=True)` for directory creation

### Error Handling
- Use `ValueError` for user-facing validation errors
- Include descriptive error messages
- Use assert for programming errors that should never occur
- Use try/except for fallback logic (e.g., optional dependencies)
- Check for model parameter existence: `if "model" not in kwargs`

### Classes
- Use `@dataclass` for simple data containers (Provider)
- Implement `__enter__`/`__exit__` for context manager support
- Initialize attributes in `__init__` even if None
- Use `Optional[str]` for potentially None attributes

### Docstrings
- Use `"""` triple quotes for module and function docstrings
- Include description and parameter/return details
- Keep docstrings concise
- Use inline comments for complex logic

### Testing
- Use pytest fixtures (e.g., tmp_path)
- Use parametrization for similar test cases
- Use `pytest.raises(ValueError, match="pattern")` for expected exceptions
- Mark slow tests with `@pytest.mark.slow`
- Mark live tests with `@pytest.mark.live`
- Test batch operations with dry_run parameter
- Test audio transcription with sample audio files

### Code Organization
- Separate utility functions into `_utils.py` module
- Use `__init__.py` to expose public API
- Keep backward compatibility in mind; deprecate old APIs with warnings
- Organize examples in `examples/` directory with clear documentation

### API Interaction
- Use `dry_run` parameter for testing without API calls
- Return mock objects/values when dry_run=True
- Pass through OpenAI client parameters (extra_headers, extra_query, extra_body, timeout)
- Use BatchType enum for batch operations
- Handle multiple batch types (chat_completion, embedding, score, rerank, transfusion, transcription)

### Constants
- Group related constants as Enums (BatchType)
- Use module-level constants for magic strings/numbers
- Use FINISHED_STATES tuple for batch completion statuses

### Validation
- Validate arguments early in functions
- Use isinstance() for type checking
- Provide specific validation error messages
- Validate request type consistency (cannot mix embedding/chat completion in same batch)
- Validate model consistency based on provider requirements

## Batch Request Types

The Batch class supports multiple request types. When adding to batch, specify appropriate parameters:

### Chat Completion
```python
batch.add_to_batch(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Embeddings
```python
batch.add_to_batch(
    model="text-embedding-3-small",
    input="text to embed"
)
```

### Score
```python
batch.add_to_batch(
    model="model-name",
    text_1="text 1",
    text_2="text 2"
)
```

### Rerank
```python
batch.add_to_batch(
    model="model-name",
    query="search query",
    documents=["doc1", "doc2"]
)
```

### Transfusion
```python
batch.add_to_batch(
    model="Shitao/OmniGen-v1",
    prompt="prompt text",
    size="512x512",
    image="data:image/...",
    response_format="url"
)
```

### Audio Transcription
```python
batch.add_to_batch(
    model="openai/whisper-large-v3",
    file=audio_data_url,  # Base64 encoded or file path
    response_format="text"
)
```

### Audio Transcription Helper
Use `data_url()` function from `openai_batch` to encode audio files:

```python
from openai_batch import data_url

audio_data_url = data_url(audio_file_path)  # Converts to base64 data URL
```

## Audio Transcription Guidelines

### Supported Audio Formats
- MP3, WAV, M4A, OGG, FLAC, MP4, WEBM
- Base64-encoded audio data URLs
- File paths to audio files

### Whisper Models
- `openai/whisper-tiny`: Fast, good accuracy
- `openai/whisper-base`: Fast, moderate accuracy
- `openai/whisper-small`: Moderate speed, very good accuracy
- `openai/whisper-medium`: Slow, excellent accuracy
- `openai/whisper-large-v2`: Slow, excellent accuracy
- `openai/whisper-large-v3`: Slow, best available accuracy

### Transcription Request Body
When processing transcription requests:
- Audio file must be provided as 'file' parameter
- For base64 data URLs: extract file_type from MIME type
- Supported parameters: model, language, response_format, temperature, prompt
- Body structure: `{model, file, file_type, language, response_format, temperature, prompt}`

### Batch Type Validation
Ensure request type matches batch type:
- Embedding requests → BatchType.EMBEDDING
- Chat completion → BatchType.CHAT_COMPLETION
- Score requests → BatchType.SCORE
- Rerank requests → BatchType.RERANK
- Transfusion requests → BatchType.TRANSFUSION
- Transcription requests → BatchType.TRANSCRIPTION

### Example Scripts
Reference examples in `examples/` directory:
- `audio_transcription.py`: Comprehensive batch audio transcription example
- `image_understanding.py`: Image analysis with batch processing

## Audio File Processing

### Base64 Encoding
Use `data_url()` helper to convert audio files to base64-encoded data URLs:
- Accepts file paths, URLs, or file-like objects
- Auto-detects MIME type
- Returns format: `data:{mime_type};base64,{encoded_data}`

### Batch Processing
- Maximum batch input size: varies by provider (OpenAI: 100MB, Parasail: 500MB)
- Maximum requests per batch: varies by provider (OpenAI: 50,000, Parasail: 50,000)
- Endpoint for transcription: `/v1/audio/transcriptions`

### Output Files
- `transcriptions.jsonl`: Successful transcription results
- `transcription_errors.jsonl`: Failed transcription attempts
- `transcripts/`: Individual text files with transcripts
- Extract text from JSONL: `response.content[].text`

## Examples and Testing

### Using Example Scripts
```bash
# Transcribe audio files
export OPENAI_API_KEY="your-key"
python examples/audio_transcription.py --audio-directory ./audio_files

# Test without API calls
python examples/audio_transcription.py --audio-directory ./audio_files --dry-run
```

### Testing Transcription
```python
# Test audio transcription
with Batch() as batch:
    audio_data_url = data_path(audio_file)
    batch.add_to_batch(
        model="openai/whisper-large-v3",
        file=audio_data_url,
        response_format="text"
    )
    result, output_path, error_path = batch.submit_wait_download()
```

### Mock Objects
When dry_run=True, batch operations return mock objects:
- Batch status: "completed"
- Mock file IDs for testing

### Resources
- Black line length: 100
- Test framework: pytest
- Main dependency: openai>=1.60
- Audio processing: ffmpeg (for transcoding if needed)
- Type stubs: Optional typing
