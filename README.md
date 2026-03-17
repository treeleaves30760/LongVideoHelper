# LongVideoHelper

A CLI tool for long video transcription with Whisper, featuring OCR-assisted vocabulary correction for game streams and domain-specific content.

## Features

- **Whisper transcription** with VAD-based clipping for long videos (1-3+ hours)
- **Vocabulary-guided correction** using LLM (local or API) with domain-specific term lists
- **OCR-assisted correction** extracts on-screen text from video frames via PaddleOCR for additional context
- **Rule-based post-correction** deterministically fixes known transcription errors
- **SRT subtitle output** for video editing software
- **Automatic retry** with fallback when Whisper encounters problematic audio segments

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

```bash
git clone git@github.com:treeleaves30760/LongVideoHelper.git
cd LongVideoHelper
uv sync
```

### Optional dependencies

```bash
# For local LLM correction (e.g., Qwen3.5-4B)
uv sync --extra transformers

# For OCR-assisted correction (PaddleOCR)
uv sync --extra ocr
```

### System requirements

- **FFmpeg**: Required for audio extraction
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt-get install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **GPU**: Recommended for Whisper and local LLM models (NVIDIA CUDA)

## Usage

### Basic transcription

```bash
uv run longvideohelper transcribe video.mp4
```

### Transcription with vocabulary correction

Create a vocabulary file (see [Vocabulary File Format](#vocabulary-file-format)) and specify a correction model:

```bash
# Using local model (Qwen3.5-4B, runs on GPU)
uv run longvideohelper transcribe video.mp4 \
  -l zh \
  --vocab-file vocab.txt \
  --correction-model transformers/Qwen/Qwen3.5-4B

# Using cloud API (Gemini, requires GEMINI_API_KEY)
uv run longvideohelper transcribe video.mp4 \
  -l zh \
  --vocab-file vocab.txt \
  --correction-model gemini/gemini-2.0-flash
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output-dir` | `output` | Output directory (creates timestamped subdirectory) |
| `-m, --model` | `turbo` | Whisper model (`tiny`, `base`, `small`, `medium`, `large`, `turbo`) |
| `-l, --language` | auto | Language code (e.g., `zh`, `en`, `ja`) |
| `--vocab-file` | — | Vocabulary file for hotwords and LLM correction |
| `--correction-model` | — | LLM model for correction (see examples above) |
| `--max-clip-duration` | 300 | Max clip length in seconds |
| `--max-clips` | all | Only process first N clips (for testing) |
| `--max-segment-duration` | 10 | Max subtitle segment length in seconds |
| `--keep-clips` | false | Keep intermediate audio clips |
| `--compute-type` | auto | CTranslate2 quantization (`float16`, `int8`, etc.) |
| `--no-hallucination-filter` | false | Disable hallucination filtering |

### Testing with fewer clips

For development and tuning, use `--max-clips` to limit processing:

```bash
# Quick test with first 2 clips (~10 min of video)
uv run longvideohelper transcribe video.mp4 -l zh \
  --vocab-file vocab.txt \
  --correction-model transformers/Qwen/Qwen3.5-4B \
  --max-clips 2
```

## Vocabulary File Format

The vocabulary file supports two types of entries:

```
# Comments start with #

# Terms: used as Whisper hotwords + LLM reference vocabulary
Once Human
異常物
畸變體
轉能電池

# Known errors: deterministic post-correction rules
# Format: wrong->correct
一場物->異常物
燃燃電池->轉能電池
葉爵->夜橘
```

See `vocab_oncehuman.txt` for a complete example.

## Output

Results are saved to `output/YYYY-MM-DD-HH-MM-SS/` with:

| File | Description |
|------|-------------|
| `*_transcript.srt` | Corrected SRT subtitles |
| `*_raw.srt` | Raw SRT (before LLM correction) |
| `*_transcript.txt` | Corrected transcript with timestamps |
| `*_transcript_plain.txt` | Plain text transcript |

The SRT files can be imported into video editors (Premiere Pro, DaVinci Resolve, Final Cut Pro, etc.).

## How Correction Works

```
Video → Extract Audio → VAD Clip → Whisper Transcribe
                                         ↓
                              Raw segments (may have errors)
                                         ↓
         Video → Extract Frames (every 10s) → PaddleOCR → Screen text
                                         ↓
         Raw segments + OCR text + Vocab → LLM Correction
                                         ↓
         Known error→correct rules → Rule-based Post-Correction
                                         ↓
                              Final corrected subtitles
```

1. **Whisper** transcribes audio with vocabulary hotwords (top 20 terms)
2. **PaddleOCR** extracts on-screen text from video frames for additional context
3. **LLM** corrects domain-specific terms using vocabulary + OCR context
4. **Rule-based** post-correction applies deterministic `wrong->correct` mappings

## Tech Stack

- **Whisper**: faster-whisper (CTranslate2) — `turbo` = `large-v3-turbo`
- **OCR**: PaddleOCR (Traditional Chinese)
- **LLM**: Local via transformers (Qwen3.5-4B) or API via LiteLLM (Gemini, OpenAI, etc.)
- **Package manager**: uv
