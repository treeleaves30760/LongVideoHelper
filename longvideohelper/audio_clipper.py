"""Module for clipping audio using Voice Activity Detection (VAD)."""

import wave
import webrtcvad
from pathlib import Path
from typing import List, Tuple, Union
from pydub import AudioSegment
import struct


class AudioClipper:
    """Handles VAD-based audio clipping to ensure clips don't cut sentences."""

    def __init__(self, max_clip_duration: int = 300, vad_aggressiveness: int = 2):
        """
        Initialize the AudioClipper.

        Args:
            max_clip_duration: Maximum duration of each clip in seconds (default: 300 = 5 minutes)
            vad_aggressiveness: WebRTC VAD aggressiveness (0-3, higher = more aggressive)
        """
        self.max_clip_duration = max_clip_duration
        self.vad = webrtcvad.Vad(vad_aggressiveness)

    def _read_wave(self, path: Union[str, Path]) -> Tuple[bytes, int, int, int]:
        """Read a WAV file and return its properties."""
        with wave.open(str(path), 'rb') as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            return frames, sample_rate, sample_width, num_channels

    def _write_wave(self, path: Union[str, Path], audio: bytes, sample_rate: int):
        """Write audio data to a WAV file."""
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

    def _frame_generator(self, frame_duration_ms: int, audio: bytes, sample_rate: int):
        """
        Generate audio frames from PCM audio data.

        Args:
            frame_duration_ms: Duration of each frame in milliseconds
            audio: PCM audio data
            sample_rate: Sample rate of the audio

        Yields:
            Audio frames
        """
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0

        while offset + n < len(audio):
            yield audio[offset:offset + n], timestamp, duration
            timestamp += duration
            offset += n

    def _vad_collector(
        self,
        sample_rate: int,
        frame_duration_ms: int,
        padding_duration_ms: int,
        audio: bytes
    ) -> List[Tuple[float, float, bytes]]:
        """
        Filter out non-voiced audio frames using VAD.

        Args:
            sample_rate: Sample rate of the audio
            frame_duration_ms: Duration of each frame in milliseconds
            padding_duration_ms: Padding to add before/after speech segments
            audio: PCM audio data

        Returns:
            List of tuples containing (start_time, end_time, audio_data) for voiced segments
        """
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        ring_buffer = []
        triggered = False
        voiced_frames = []
        voiced_segments = []

        start_time = 0.0

        for frame, timestamp, duration in self._frame_generator(frame_duration_ms, audio, sample_rate):
            is_speech = self.vad.is_speech(frame, sample_rate)

            if not triggered:
                ring_buffer.append((frame, timestamp))
                if len(ring_buffer) > num_padding_frames:
                    ring_buffer.pop(0)

                num_voiced = len([f for f, t in ring_buffer if self.vad.is_speech(f, sample_rate)])

                if num_voiced > 0.9 * num_padding_frames:
                    triggered = True
                    start_time = ring_buffer[0][1]
                    for f, t in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer = []
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, timestamp))

                if len(ring_buffer) > num_padding_frames:
                    ring_buffer.pop(0)

                num_unvoiced = len([f for f, t in ring_buffer if not self.vad.is_speech(f, sample_rate)])

                if num_unvoiced > 0.9 * num_padding_frames:
                    triggered = False
                    end_time = timestamp + duration
                    voiced_segments.append((start_time, end_time, b''.join(voiced_frames)))
                    ring_buffer = []
                    voiced_frames = []

        # Handle any remaining voiced frames
        if voiced_frames:
            end_time = timestamp + duration
            voiced_segments.append((start_time, end_time, b''.join(voiced_frames)))

        return voiced_segments

    def clip_audio(
        self,
        audio_path: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> List[Tuple[Path, float, float]]:
        """
        Clip audio into segments using VAD, ensuring no sentence is cut.

        Args:
            audio_path: Path to the input audio file (WAV format)
            output_dir: Directory to save the audio clips

        Returns:
            List of tuples containing (clip_path, start_time, end_time)
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing audio: {audio_path}")

        # Convert to mono WAV at 16kHz if needed (required for webrtcvad)
        audio = AudioSegment.from_wav(str(audio_path))
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

        # Save temporary processed audio
        temp_path = output_dir / "temp_processed.wav"
        audio.export(str(temp_path), format="wav")

        # Read the processed audio
        frames, sample_rate, sample_width, num_channels = self._read_wave(temp_path)

        # Get voiced segments using VAD
        frame_duration_ms = 30  # 30ms frames
        padding_duration_ms = 300  # 300ms padding
        voiced_segments = self._vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, frames)

        print(f"Found {len(voiced_segments)} voiced segments")

        # Combine segments into clips that are under max_clip_duration
        clips = []
        current_clip_data = []
        current_clip_start = 0.0
        current_duration = 0.0

        for i, (start_time, end_time, audio_data) in enumerate(voiced_segments):
            segment_duration = end_time - start_time

            # If adding this segment would exceed max duration, save current clip
            if current_duration + segment_duration > self.max_clip_duration and current_clip_data:
                clip_path = output_dir / f"clip_{len(clips):04d}.wav"
                self._write_wave(clip_path, b''.join(current_clip_data), sample_rate)
                clips.append((clip_path, current_clip_start, current_clip_start + current_duration))

                # Start new clip
                current_clip_data = [audio_data]
                current_clip_start = start_time
                current_duration = segment_duration
            else:
                # Add segment to current clip
                if not current_clip_data:
                    current_clip_start = start_time
                current_clip_data.append(audio_data)
                current_duration += segment_duration

        # Save the last clip
        if current_clip_data:
            clip_path = output_dir / f"clip_{len(clips):04d}.wav"
            self._write_wave(clip_path, b''.join(current_clip_data), sample_rate)
            clips.append((clip_path, current_clip_start, current_clip_start + current_duration))

        # Clean up temporary file
        temp_path.unlink()

        print(f"Created {len(clips)} audio clips")
        return clips
