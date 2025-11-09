"""Utility functions for LongVideoHelper."""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from moviepy import VideoFileClip


def format_timestamp(seconds: float) -> str:
    """
    Format seconds as HH:MM:SS or MM:SS.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def parse_timestamp(timestamp_str: str) -> float:
    """
    Parse a timestamp string (HH:MM:SS or MM:SS) to seconds.

    Args:
        timestamp_str: Timestamp string

    Returns:
        Time in seconds
    """
    parts = timestamp_str.split(':')

    if len(parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:  # MM:SS
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")


def parse_transcript_file(file_path: Path) -> Dict:
    """
    Parse a transcript file with timestamps.

    Expected format: [start - end] text

    Args:
        file_path: Path to transcript file

    Returns:
        Dictionary with 'segments' list and 'text' string
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    segments = []
    full_text = []

    # Pattern to match: [0.00 - 5.23] text
    pattern = r'\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s*(.+)'

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = re.match(pattern, line)
            if match:
                start_time = float(match.group(1))
                end_time = float(match.group(2))
                text = match.group(3).strip()

                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'text': text
                })

                full_text.append(text)

    return {
        'segments': segments,
        'text': ' '.join(full_text)
    }


def get_video_duration(video_path: Path) -> float:
    """
    Get the duration of a video file in seconds.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = VideoFileClip(str(video_path))
    duration = video.duration
    video.close()

    return duration


def validate_timerange(start: float, end: float, duration: float) -> bool:
    """
    Validate a time range against video duration.

    Args:
        start: Start time in seconds
        end: End time in seconds
        duration: Total video duration in seconds

    Returns:
        True if valid, False otherwise
    """
    if start < 0 or end < 0:
        return False

    if start >= end:
        return False

    if end > duration:
        return False

    return True


def get_segments_in_timerange(
    segments: List[Dict],
    start_time: float,
    end_time: float
) -> List[Dict]:
    """
    Filter transcript segments that fall within a time range.

    Args:
        segments: List of segment dictionaries with 'start', 'end', 'text'
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        List of segments within the time range
    """
    filtered = []

    for segment in segments:
        seg_start = segment.get('start', 0)
        seg_end = segment.get('end', 0)

        # Check if segment overlaps with the time range
        if seg_end > start_time and seg_start < end_time:
            filtered.append(segment)

    return filtered


def format_duration(seconds: float) -> str:
    """
    Format duration in a human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Human-readable duration string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return ' '.join(parts)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be safe for use as a filename.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip('. ')

    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]

    return sanitized or 'unnamed'
