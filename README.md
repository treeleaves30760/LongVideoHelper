# LongVideoHelper

This project aim to offer a tool that use whisper and VLM for long video chapter dividing.

## Workflow

User input a Video, and we will create the pure audio (mp4). While processing the audio, since the video may up to 3 hours, the clip algorithm is we will first clip the video with webrtcvad, make sure each clip will not cut one sentence, than I want each clip is smaller than 5 minutes because whisper will perform more accuracy in this length. Once we get a clip, we will use the whisper to transcribe.

After whole video finish the transcribe, than we will try to ask VLM to seperate the video into several chapter. Since the transcribe result may not be correct, so we need the aid of the video. First we will send the whole transcript, and ask the VLM to output the detailed chapter start and end time, each chapter should control in 6 minutes. Then for each chapter, we will get the key frames with transcribe, and ask LLM to correct the transcribe while output the summary for this chapter.

We expected output is a full-video transcript and the summary md for each chapter information.

## Technuqie stack

- Whisper turbo
- VLM:
  - Cloud: Gemini flash 2.5
  - Local: Gemma3-4B or Gemma3-12B, Qwen3-VL-8B
  - LLMInterface: create a llm.py, and use the LiteLLM to allow using cloud or ollama etc.
- Interface: CLI, ready for GUI(TKinter in future)
- Use uv.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

1. Clone the repository:
```bash
git clone <repository-url>
cd LongVideoHelper
```

2. Install dependencies:
```bash
uv sync
```

3. Ensure you have FFmpeg installed on your system:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

### Transcribe a Video

Transcribe a video file using Whisper with automatic VAD-based clipping:

```bash
uv run longvideohelper transcribe path/to/video.mp4
```

Options:
- `-o, --output-dir PATH`: Output directory for results (default: `output`)
- `-m, --model MODEL`: Whisper model to use (tiny, base, small, medium, large, turbo; default: turbo)
- `-l, --language LANG`: Language code (e.g., en, zh) or auto-detect if not specified
- `--max-clip-duration SECONDS`: Maximum duration of each clip in seconds (default: 300)
- `--keep-clips`: Keep intermediate audio clips (default: delete after transcription)

Example with options:
```bash
uv run longvideohelper transcribe video.mp4 -o results -m turbo -l en --keep-clips
```

### Transcribe Audio Only

Transcribe an audio file directly without VAD clipping:

```bash
uv run longvideohelper transcribe-audio path/to/audio.wav
```

Options:
- `-o, --output-dir PATH`: Output directory for results
- `-m, --model MODEL`: Whisper model to use
- `-l, --language LANG`: Language code or auto-detect

### Using Python API

You can also use the modules programmatically:

```python
from longvideohelper.audio_extractor import AudioExtractor
from longvideohelper.audio_clipper import AudioClipper
from longvideohelper.transcriber import Transcriber

# Extract audio from video
extractor = AudioExtractor()
audio_path = extractor.extract_audio("video.mp4")

# Clip audio using VAD
clipper = AudioClipper(max_clip_duration=300)
clips = clipper.clip_audio(audio_path, "output/clips")

# Transcribe clips
transcriber = Transcriber(model_name="turbo")
transcriptions = transcriber.transcribe_clips(clips)
merged = transcriber.merge_transcriptions(transcriptions)
transcriber.save_transcription(merged, "output/transcript.txt")
```

## Current Status

✅ **Phase 1: Video to Transcribe (COMPLETED)**
- Audio extraction from video
- VAD-based audio clipping
- Whisper transcription
- CLI interface

🚧 **Phase 2: VLM Chapter Division (TODO)**
- Chapter detection using VLM
- Key frame extraction
- Transcript correction with VLM
- Chapter summaries

## Output Format

The tool generates two files:
1. `{video_name}_transcript.txt`: Full transcript with timestamps
2. `{video_name}_transcript_plain.txt`: Plain text transcript without timestamps

Example transcript with timestamps:
```
[0.00 - 5.23] Welcome to this video tutorial.
[5.23 - 12.45] Today we'll be discussing Python programming.
...
```
