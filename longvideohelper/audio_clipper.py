"""Module for splitting audio into clips at silence boundaries."""

from pathlib import Path
from typing import List, Tuple, Union
from pydub import AudioSegment
from pydub.silence import detect_silence


class AudioClipper:
    """Splits audio into clips at natural silence boundaries, preserving all original audio."""

    def __init__(self, max_clip_duration: int = 300, silence_thresh: int = -40, min_silence_len: int = 500):
        """
        Initialize the AudioClipper.

        Args:
            max_clip_duration: Maximum duration of each clip in seconds (default: 300 = 5 minutes)
            silence_thresh: Silence threshold in dBFS (default: -40)
            min_silence_len: Minimum silence length in ms to be considered a split point (default: 500)
        """
        self.max_clip_duration = max_clip_duration
        self.silence_thresh = silence_thresh
        self.min_silence_len = min_silence_len

    def clip_audio(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> List[Tuple[Path, float, float]]:
        """
        Split audio into clips at silence boundaries, preserving all original audio.

        Args:
            audio_path: Path to the input audio file (WAV format)
            output_dir: Directory to save the audio clips

        Returns:
            List of tuples containing (clip_path, start_time_sec, end_time_sec)
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing audio: {audio_path}")

        # Load and convert to mono 16kHz 16-bit (optimal for Whisper)
        audio = AudioSegment.from_wav(str(audio_path))
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        total_duration_ms = len(audio)
        max_clip_ms = self.max_clip_duration * 1000

        print(f"Audio duration: {total_duration_ms / 1000:.1f}s")

        # If audio fits in one clip, just export it directly
        if total_duration_ms <= max_clip_ms:
            clip_path = output_dir / "clip_0000.wav"
            audio.export(str(clip_path), format="wav")
            print("Created 1 audio clip (entire file)")
            return [(clip_path, 0.0, total_duration_ms / 1000.0)]

        # Find silence ranges: list of [start_ms, end_ms]
        silences = detect_silence(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh
        )
        # Use midpoints of silence ranges as split candidates
        split_candidates = [(s + e) // 2 for s, e in silences]

        print(f"Found {len(split_candidates)} silence-based split candidates")

        # Build split points near max_clip_duration boundaries
        split_points = []
        cursor = 0
        while cursor + max_clip_ms < total_duration_ms:
            target = cursor + max_clip_ms
            # Find the silence midpoint closest to target
            best = None
            best_dist = float('inf')
            for sp in split_candidates:
                if sp <= cursor:
                    continue
                dist = abs(sp - target)
                if dist < best_dist:
                    best = sp
                    best_dist = dist
                # Stop searching once we're far past the target
                if sp > target + max_clip_ms // 2:
                    break

            if best is not None and best > cursor:
                split_points.append(best)
                cursor = best
            else:
                # No good silence found; hard-split at max duration
                split_points.append(target)
                cursor = target

        # Create clips from split points
        boundaries = [0] + split_points + [total_duration_ms]
        clips = []
        for i in range(len(boundaries) - 1):
            start_ms = boundaries[i]
            end_ms = boundaries[i + 1]
            clip = audio[start_ms:end_ms]
            clip_path = output_dir / f"clip_{i:04d}.wav"
            clip.export(str(clip_path), format="wav")
            clips.append((clip_path, start_ms / 1000.0, end_ms / 1000.0))

        print(f"Created {len(clips)} audio clips")
        return clips
