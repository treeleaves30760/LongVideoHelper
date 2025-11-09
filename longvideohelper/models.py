"""Data models for LongVideoHelper."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional


@dataclass
class Chapter:
    """Represents a chapter in a video."""

    index: int
    start_time: float
    end_time: float
    title: Optional[str] = None
    transcript_segments: List[Dict] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get chapter duration in seconds."""
        return self.end_time - self.start_time

    def get_transcript_text(self) -> str:
        """Get the transcript text for this chapter."""
        return ' '.join([seg.get('text', '').strip() for seg in self.transcript_segments])

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'Chapter':
        """Create Chapter from dictionary."""
        return cls(**data)


@dataclass
class ChapterResult:
    """Represents the result of processing a chapter with VLM."""

    chapter: Chapter
    corrected_transcript: str
    summary: str
    key_points: List[str] = field(default_factory=list)
    keyframes: List[Path] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if processing was successful."""
        return self.error is None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        data = {
            'chapter': self.chapter.to_dict(),
            'corrected_transcript': self.corrected_transcript,
            'summary': self.summary,
            'key_points': self.key_points,
            'keyframes': [str(kf) for kf in self.keyframes],
            'error': self.error
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChapterResult':
        """Create ChapterResult from dictionary."""
        chapter_data = data.pop('chapter')
        chapter = Chapter.from_dict(chapter_data)

        keyframes = [Path(kf) for kf in data.pop('keyframes', [])]

        return cls(
            chapter=chapter,
            keyframes=keyframes,
            **data
        )


@dataclass
class VideoMetadata:
    """Metadata about a video file."""

    path: Path
    duration: float
    language: str = 'unknown'
    total_chapters: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'path': str(self.path),
            'duration': self.duration,
            'language': self.language,
            'total_chapters': self.total_chapters
        }
