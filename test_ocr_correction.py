"""Dev test script: run transcription on pre-generated clips for fast iteration.

Usage:
    python test_ocr_correction.py [MAX_CLIPS]

Prerequisites:
    1. Generate clips once:
       python -c "
       from longvideohelper.audio_extractor import AudioExtractor
       from longvideohelper.audio_clipper import AudioClipper
       AudioExtractor().extract_audio('path/to/video.mp4', 'output/audio.wav')
       AudioClipper().clip_audio('output/audio.wav', 'output/clips')
       "
    2. Then run this script to test transcription + correction without re-extracting audio.
"""
import sys
import logging
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)

VIDEO_PATH = r"T:\Video\2026-03-12 19-41-29 (Once Human).mp4"
CLIPS_DIR = Path("output/clips")
VOCAB_FILE = "vocab_oncehuman.txt"
CORRECTION_MODEL = "transformers/Qwen/Qwen3.5-4B"
MAX_CLIPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2

output_dir = Path("output") / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
output_dir.mkdir(parents=True, exist_ok=True)

clip_files = sorted(CLIPS_DIR.glob("clip_*.wav"))
if not clip_files:
    print("No clips found in output/clips/ — generate them first (see docstring).")
    sys.exit(1)

from pydub import AudioSegment
clips = []
offset = 0.0
for f in clip_files[:MAX_CLIPS]:
    dur = len(AudioSegment.from_wav(str(f))) / 1000.0
    clips.append((f, offset, offset + dur))
    offset += dur

print(f"Testing {len(clips)}/{len(clip_files)} clips", flush=True)
print(f"Output: {output_dir}", flush=True)

from longvideohelper.transcriber import Transcriber

transcriber = Transcriber(
    model_name="turbo",
    vocab_file=VOCAB_FILE,
    correction_model=CORRECTION_MODEL,
    video_path=VIDEO_PATH,
    max_segment_duration=10,
)

results = transcriber.transcribe_clips(clips, language="zh")

merged = transcriber.merge_transcriptions(results)
raw_merged = transcriber.merge_transcriptions([
    {**t, "segments": t.get("raw_segments", t.get("segments", []))}
    for t in results
])

stem = Path(VIDEO_PATH).stem
transcriber.save_transcription_srt(merged, output_dir / f"{stem}_transcript.srt")
transcriber.save_transcription_srt(raw_merged, output_dir / f"{stem}_raw.srt")
transcriber.save_transcription(merged, output_dir / f"{stem}_transcript.txt", include_timestamps=True)
transcriber.save_transcription(raw_merged, output_dir / f"{stem}_raw.txt", include_timestamps=True)
transcriber.save_transcription(merged, output_dir / f"{stem}_transcript_plain.txt", include_timestamps=False)

print(f"\nResults: {output_dir}/", flush=True)

raw_segs = raw_merged.get("segments", [])
cor_segs = merged.get("segments", [])
changes = [(i, raw_segs[i]["text"].strip(), cor_segs[i]["text"].strip())
           for i in range(min(len(raw_segs), len(cor_segs)))
           if raw_segs[i]["text"].strip() != cor_segs[i]["text"].strip()]

print(f"Segments: {len(raw_segs)}, Corrections: {len(changes)}", flush=True)
if changes:
    for idx, r, c in changes[:10]:
        print(f"  [{idx}] {r[:60]}", flush=True)
        print(f"    -> {c[:60]}", flush=True)
    if len(changes) > 10:
        print(f"  ... and {len(changes)-10} more", flush=True)
