"""Module for extracting audio from video files."""

from pathlib import Path
from moviepy import VideoFileClip
from typing import Optional, Union


class AudioExtractor:
    """Handles extraction of audio from video files."""

    def __init__(self):
        """Initialize the AudioExtractor."""
        pass

    def extract_audio(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        audio_format: str = "wav"
    ) -> Path:
        """
        Extract audio from a video file.

        Args:
            video_path: Path to the input video file
            output_path: Path for the output audio file (optional)
            audio_format: Format for the output audio (default: "wav")

        Returns:
            Path to the extracted audio file

        Raises:
            FileNotFoundError: If the video file doesn't exist
            ValueError: If the video file is invalid
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Generate output path if not provided
        if output_path is None:
            output_path = video_path.with_suffix(f".{audio_format}")
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Extracting audio from: {video_path}")
        print(f"Output: {output_path}")

        # Extract audio using moviepy
        try:
            video = VideoFileClip(str(video_path))
            audio = video.audio

            if audio is None:
                raise ValueError(f"No audio track found in video: {video_path}")

            # Write audio file
            audio.write_audiofile(
                str(output_path),
                codec='pcm_s16le' if audio_format == 'wav' else None
            )

            # Close video to free resources
            video.close()

            print(f"Audio extracted successfully to: {output_path}")
            return output_path

        except Exception as e:
            raise ValueError(f"Failed to extract audio from video: {str(e)}")
