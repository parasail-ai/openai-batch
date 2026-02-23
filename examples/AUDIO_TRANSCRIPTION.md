# Batch Audio Transcription Examples

This directory contains examples for batch audio transcription using the OpenAI Batch SDK.

## Available Scripts

### 1. `audio_transcription.py` - Simple Example
Basic batch transcription example with minimal configuration.

**Usage:**
```bash
# Transcribe audio files
python audio_transcription.py --audio-directory ./audio_files

# Use different whisper model
python audio_transcription.py --audio-directory ./audio_files --model openai/whisper-tiny

# Test without making API calls
python audio_transcription.py --audio-directory ./audio_files --dry-run
```

### 2. `batch_transcribe.py` - Comprehensive Example
Feature-rich example with advanced options and result processing.

**Usage:**
```bash
# Basic transcription
python batch_transcribe.py --audio-directory ./audio_files

# Custom model and output files
python batch_transcribe.py \
  --audio-directory ./audio_files \
  --model openai/whisper-large-v2 \
  --output-file my_transcriptions.jsonl

# Process multiple directories
python batch_transcribe.py \
  --audio-directory ./podcasts \
  --audio-directory ./videos \
  --output-file combined_transcriptions.jsonl
```

## Prerequisites

### Required
- OpenAI API key (export: `export OPENAI_API_KEY="your-key-here"`)

### Optional
- Parasail API key (export: `export PARASAIL_API_KEY="your-key-here"`) for different models

### Audio Files
Supported formats: MP3, WAV, M4A, OGG, FLAC, MP4, WEBM

## How It Works

### Basic Workflow
1. **Find Audio Files**: Script searches for audio files in the specified directory
2. **Create Batch**: Uses OpenAI Batch SDK to create batch configuration
3. **Transcribe**: Submits batch to Whisper API for transcription
4. **Download Results**: Retrieves transcriptions from API
5. **Process Results**: Displays and saves individual transcripts

### Features
- **Batch Processing**: Transcribe thousands of audio files efficiently
- **Multiple Formats**: Support for various audio and video formats
- **Model Selection**: Choose from Whisper models (tiny, base, small, medium, large)
- **Progress Tracking**: Real-time progress updates
- **Error Handling**: Graceful handling of failed transcriptions
- **Result Extraction**: Save individual transcripts to text files

## Output Files

### `transcriptions.jsonl`
Line-delimited JSON file with successful transcription results:
```json
{"custom_id": "line-1", "status": "success", "response": {"content": [{"text": "Hello world"}]}}

{"custom_id": "line-2", "status": "success", "response": {"content": [{"text": "Sample audio transcript"}]}}
```

### `transcription_errors.jsonl`
Line-delimited JSON file with failed transcription attempts:
```json
{"custom_id": "line-3", "status": "failed", "error": "API error: file too large"}
```

### `transcripts/`
Individual text files for each successful transcription:
```
transcripts/
├── transcript_line_1.txt
├── transcript_line_2.txt
├── transcript_line_3.txt
└── ...
```

## Whisper Models

| Model | Speed | Accuracy | Best Use Case |
|-------|-------|----------|---------------|
| `openai/whisper-tiny` | Fast | Good | Real-time processing, testing |
| `openai/whisper-base` | Fast | Good | Everyday use, moderate accuracy |
| `openai/whisper-small` | Moderate | Very Good | General transcription |
| `openai/whisper-medium` | Slow | Excellent | Professional use |
| `openai/whisper-large-v2` | Slow | Excellent | High accuracy needs |
| `openai/whisper-large-v3` | Slow | Best Available | Production quality |

## API Key Selection

The script automatically detects the appropriate provider:
- `openai/whisper-*` models → OpenAI provider (requires `OPENAI_API_KEY`)
- Other models → Parasail provider (requires `PARASAIL_API_KEY`)

## Examples

### Example 1: Transcribe Podcast Episodes
```bash
export OPENAI_API_KEY="sk-..."
python batch_transcribe.py --audio-directory ./podcast_episodes
```

### Example 2: Test Without API Calls
```bash
python audio_transcription.py --audio-directory ./test_audio --dry-run
```

### Example 3: Use Faster Model
```bash
export OPENAI_API_KEY="sk-..."
python batch_transcribe.py \
  --audio-directory ./large_collection \
  --model openai/whisper-small
```

### Example 4: Process Video Files
```bash
export OPENAI_API_KEY="sk-..."
python batch_transcribe.py \
  --audio-directory ./meetings
```

## Understanding Batch Processing

### What is Batch Processing?
Batch processing allows you to submit many transcription requests at once, which is much more efficient than submitting them one at a time.

### Benefits
- **Cost Effective**: Batch processing is often cheaper than individual API calls
- **Faster**: All files processed in parallel
- **Scalable**: Handle thousands of files with a single script
- **Resumable**: Check status and resume if needed

### Batch Lifecycle
1. **Create**: Build batch configuration with audio files
2. **Submit**: Send batch to OpenAI API
3. **Process**: API processes files in parallel
4. **Complete**: Receive results when ready
5. **Download**: Extract transcription data

## Troubleshooting

### No audio files found
- Check that your directory path is correct
- Verify audio files have supported extensions (.mp3, .wav, etc.)
- Use absolute paths if needed

### API Key not found
- Ensure you've exported the API key: `export OPENAI_API_KEY="your-key"`
- Check with: `echo $OPENAI_API_KEY`

### Batch submission failed
- Verify API key is valid
- Check internet connection
- Review API rate limits
- Try smaller batch size

### Processing taking too long
- Batch processing is asynchronous
- Script polls for completion status
- Check console output for progress updates
- Larger files take more time

### Results incomplete
- Check `transcription_errors.jsonl` for specific failures
- Batch may still be processing
- Use dry-run to test before running full batch

## Additional Resources

- [OpenAI Batch API Documentation](https://platform.openai.com/docs/api-reference/batch)
- [Whisper Model Documentation](https://platform.openai.com/docs/models/whisper)
- [OpenAI Batch Cookbook](https://cookbook.openai.com/examples/batch_processing)
- [OpenAI API Key Management](https://platform.openai.com/account/api-keys)
