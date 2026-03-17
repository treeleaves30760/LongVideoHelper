# LongVideoHelper

A CLI tool for long video transcription and chapter segmentation, featuring OCR-assisted vocabulary correction for game streams and domain-specific content.

## Features

- **Whisper transcription** with VAD-based clipping for long videos (1-3+ hours)
- **Vocabulary-guided correction** using LLM (local or API) with domain-specific term lists
- **OCR-assisted correction** extracts on-screen text from video frames via PaddleOCR
- **Chapter segmentation** automatically divides videos into titled chapters with summaries using LLM
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

## Commands

### `transcribe` — Video transcription

Extracts audio, clips with VAD, transcribes with Whisper, and optionally detects chapters.

```bash
# Basic transcription
uv run longvideohelper transcribe video.mp4

# With vocabulary correction
uv run longvideohelper transcribe video.mp4 \
  -l zh \
  --vocab-file vocab.txt \
  --correction-model gemini/gemini-2.0-flash

# With vocabulary correction + chapter detection
uv run longvideohelper transcribe video.mp4 \
  -l zh \
  --vocab-file vocab.txt \
  --correction-model ollama/gpt-oss:20b \
  --chapters
```

#### Transcribe options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output-dir` | `output` | Output directory (creates timestamped subdirectory) |
| `-m, --model` | `turbo` | Whisper model (`tiny`, `base`, `small`, `medium`, `large`, `turbo`) |
| `-l, --language` | auto | Language code (e.g., `zh`, `en`, `ja`) |
| `--vocab-file` | — | Vocabulary file for hotwords and LLM correction |
| `--correction-model` | — | LLM model for correction (e.g., `gemini/gemini-2.0-flash`, `ollama/qwen3:8b`, `transformers/Qwen/Qwen3.5-4B`) |
| `--chapters` | false | Enable chapter detection after transcription |
| `--chapter-model` | — | LLM model for chapters (defaults to `--correction-model`) |
| `--chapter-duration` | 300 | Target chapter duration in seconds |
| `--max-clip-duration` | 300 | Max audio clip length in seconds |
| `--max-clips` | all | Only process first N clips (for testing) |
| `--max-segment-duration` | 10 | Max subtitle segment length in seconds |
| `--keep-clips` | false | Keep intermediate audio clips |
| `--compute-type` | auto | CTranslate2 quantization (`float16`, `int8`, etc.) |
| `--no-hallucination-filter` | false | Disable hallucination filtering |

### `detect-chapters` — Chapter detection from existing transcript

Detects chapter boundaries from an existing transcript file without re-transcribing. Useful for iterating on chapter settings or using a different model.

```bash
# Detect chapters from a transcript
uv run longvideohelper detect-chapters transcript.txt -m ollama/gpt-oss:20b

# With custom chapter duration
uv run longvideohelper detect-chapters transcript.txt \
  -m gemini/gemini-2.0-flash \
  --chapter-duration 600 \
  --min-chapter-duration 180 \
  --max-chapter-duration 1200
```

#### Detect-chapters options

| Option | Default | Description |
|--------|---------|-------------|
| `-m, --model` | (required) | LLM model for chapter detection |
| `-o, --output-dir` | same as transcript | Output directory |
| `-d, --duration` | auto | Total video duration in seconds |
| `--chapter-duration` | 300 | Target chapter duration in seconds |
| `--min-chapter-duration` | 120 | Minimum chapter duration in seconds |
| `--max-chapter-duration` | 900 | Maximum chapter duration in seconds |

### `transcribe-audio` — Direct audio transcription

Transcribes an audio file directly without VAD clipping.

```bash
uv run longvideohelper transcribe-audio audio.wav -l zh
```

### `create-chapters` — Chapter creation with VLM + keyframes

Advanced chapter creation using a Vision-Language Model with keyframe extraction. Produces markdown summaries with embedded screenshots.

```bash
uv run longvideohelper create-chapters video.mp4 \
  -t transcript.txt \
  --vlm-provider gemini \
  --vlm-model gemini-2.0-flash-exp
```

### `process` — Full pipeline (transcribe + VLM chapters)

Combines `transcribe` and `create-chapters` in one command.

```bash
uv run longvideohelper process video.mp4 \
  --vlm-provider gemini \
  --vlm-model gemini-2.0-flash-exp
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

Results are saved to `output/YYYY-MM-DD-HH-MM-SS/`:

### Transcription output

| File | Description |
|------|-------------|
| `*_transcript.srt` | Corrected SRT subtitles |
| `*_raw.srt` | Raw SRT (before LLM correction) |
| `*_transcript.txt` | Corrected transcript with timestamps |
| `*_transcript_plain.txt` | Plain text transcript |

### Chapter output

| File | Description |
|------|-------------|
| `*_chapters.txt` | Chapter list with YouTube-style timestamps and summaries |
| `*_chapters.json` | Structured chapter data (start/end times, titles, summaries) |

The `_chapters.txt` file includes a ready-to-paste YouTube chapter timeline:

```
## 時間軸

0:00:06 開場與裝備規劃
0:05:10 伺服器選擇與初始任務
0:10:13 戰鬥與異常生物收集
0:15:01 建築與科技升級
...
```

## How It Works

### Transcription pipeline

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

### Chapter detection

For videos longer than 30 minutes, transcripts are automatically split into ~30-minute chunks with 2-minute overlap. Each chunk is processed independently for better accuracy, then results are merged.

```
Corrected transcript
        ↓
Condense into time-windowed format (15s-120s adaptive)
        ↓
Split into ~30 min chunks (with overlap)
        ↓
LLM detects chapter boundaries per chunk
        ↓
Merge chapters across chunks → Validate → Output
```

## Project Structure

```
longvideohelper/
├── main.py               # CLI entry point (Click commands)
├── transcriber.py         # Whisper transcription + LLM correction pipeline
├── chapter_segmenter.py   # LLM-based chapter detection (litellm backend)
├── audio_extractor.py     # Video → WAV audio extraction (FFmpeg)
├── audio_clipper.py       # VAD-based audio clipping (silence detection)
├── llm.py                 # VLMClient for vision-language model access
├── chapter_detector.py    # VLM-based chapter boundary detection
├── chapter_processor.py   # Per-chapter VLM processing with keyframes
├── keyframe_extractor.py  # Video frame extraction (OpenCV + MoviePy)
├── markdown_generator.py  # Markdown/JSON output for VLM chapters
├── models.py              # Data models (Chapter, ChapterResult, VideoMetadata)
└── utils.py               # Timestamp formatting, transcript parsing, helpers
```

### Key modules

- **`transcriber.py`** — Core transcription engine. Handles Whisper inference with automatic retry/fallback, segment splitting at punctuation boundaries, LLM correction with batching, OCR frame extraction, and cross-clip context propagation.
- **`chapter_segmenter.py`** — Lightweight chapter detection via litellm. Condenses transcripts into time windows, automatically chunks long videos, calls LLM for boundary detection with title/summary generation, and validates results.
- **`audio_clipper.py`** — Splits long audio into ~5-minute clips at natural silence boundaries using VAD (Voice Activity Detection) with -40dBFS threshold.

## Supported Models

### Whisper models

`tiny`, `base`, `small`, `medium`, `large`, `turbo` (= large-v3-turbo)

### LLM models (for correction and chapter detection)

Any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers):

| Provider | Example | Notes |
|----------|---------|-------|
| Gemini | `gemini/gemini-2.0-flash` | Requires `GEMINI_API_KEY` |
| OpenAI | `openai/gpt-4o-mini` | Requires `OPENAI_API_KEY` |
| Ollama | `ollama/gpt-oss:20b` | Local, no API key needed |
| Transformers | `transformers/Qwen/Qwen3.5-4B` | Local GPU, correction only |

## Tech Stack

- **Whisper**: faster-whisper (CTranslate2) — `turbo` = `large-v3-turbo`
- **OCR**: PaddleOCR (Traditional Chinese)
- **LLM**: Local via transformers or Ollama, cloud via LiteLLM (Gemini, OpenAI, etc.)
- **CLI**: Click
- **Package manager**: uv
