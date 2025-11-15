"""Module for transcribing audio using Whisper."""

import whisper
from pathlib import Path
from typing import Dict, List, Union, Optional
from tqdm import tqdm


class Transcriber:
    """Handles audio transcription using OpenAI's Whisper model."""

    def __init__(self, model_name: str = "turbo", device: Optional[str] = None):
        """
        Initialize the Transcriber.

        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large, turbo)
            device: Device to run the model on (cuda, cpu, or None for auto)
        """
        self.model_name = model_name
        print(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name, device=device)
        print(f"Model loaded successfully")

    def transcribe_clip(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe a single audio clip.

        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en', 'zh') or None for auto-detect

        Returns:
            Dictionary containing transcription results with keys:
            - text: The transcribed text
            - segments: List of segment dictionaries with timing information
            - language: Detected language
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Transcribe using Whisper
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            verbose=False
        )

        return result

    def transcribe_clips(
        self,
        clips: List[tuple],
        language: Optional[str] = None
    ) -> List[Dict]:
        """
        Transcribe multiple audio clips.

        Args:
            clips: List of tuples (clip_path, start_time, end_time)
            language: Language code or None for auto-detect

        Returns:
            List of transcription results, each containing:
            - clip_path: Path to the audio clip
            - start_time: Start time in the original audio
            - end_time: End time in the original audio
            - text: Transcribed text
            - segments: Detailed segment information
            - language: Detected language
        """
        results = []

        print(f"\nTranscribing {len(clips)} audio clips...")

        for clip_path, start_time, end_time in tqdm(clips, desc="Transcribing"):
            try:
                transcription = self.transcribe_clip(clip_path, language)

                # Adjust segment timestamps to reflect position in original audio
                adjusted_segments = []
                for segment in transcription.get('segments', []):
                    adjusted_segment = segment.copy()
                    adjusted_segment['start'] += start_time
                    adjusted_segment['end'] += start_time
                    adjusted_segments.append(adjusted_segment)

                result = {
                    'clip_path': clip_path,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': transcription['text'],
                    'segments': adjusted_segments,
                    'language': transcription.get('language', 'unknown')
                }

                results.append(result)

            except Exception as e:
                print(f"\nError transcribing {clip_path}: {str(e)}")
                # Add empty result to maintain order
                results.append({
                    'clip_path': clip_path,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': '',
                    'segments': [],
                    'language': 'unknown',
                    'error': str(e)
                })

        return results

    def merge_transcriptions(self, transcriptions: List[Dict]) -> Dict:
        """
        Merge multiple transcriptions into a single result.

        Args:
            transcriptions: List of transcription results

        Returns:
            Merged transcription with:
            - text: Full transcription text
            - segments: All segments combined
            - language: Most common language detected
        """
        full_text = ' '.join([t['text'].strip() for t in transcriptions if t['text']])

        all_segments = []
        for t in transcriptions:
            all_segments.extend(t.get('segments', []))

        # Get most common language
        languages = [t['language'] for t in transcriptions if 'error' not in t]
        language = max(set(languages), key=languages.count) if languages else 'unknown'

        return {
            'text': full_text,
            'segments': all_segments,
            'language': language
        }

    def save_transcription(
        self,
        transcription: Dict,
        output_path: Union[str, Path],
        include_timestamps: bool = True
    ):
        """
        Save transcription to a file.

        Args:
            transcription: Transcription result dictionary
            output_path: Path to save the transcription
            include_timestamps: Whether to include timestamps in the output
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            if include_timestamps and 'segments' in transcription:
                # Write with timestamps
                for segment in transcription['segments']:
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '').strip()
                    f.write(f"[{start:.2f} - {end:.2f}] {text}\n")
            else:
                # Write plain text
                f.write(transcription['text'])

        print(f"Transcription saved to: {output_path}")

    def save_transcription_srt(
        self,
        transcription: Dict,
        output_path: Union[str, Path]
    ):
        """
        Save transcription in SRT (SubRip) format for video editing.

        Args:
            transcription: Transcription result dictionary with segments
            output_path: Path to save the SRT file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            if 'segments' in transcription:
                for i, segment in enumerate(transcription['segments'], start=1):
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '').strip()

                    # Convert seconds to SRT timestamp format (HH:MM:SS,mmm)
                    start_time = self._seconds_to_srt_time(start)
                    end_time = self._seconds_to_srt_time(end)

                    # Write SRT entry
                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

        print(f"SRT file saved to: {output_path}")

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """
        Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
