"""Module for extracting keyframes from video files."""

import cv2
from pathlib import Path
from typing import List, Union, Tuple
from moviepy import VideoFileClip
import numpy as np


class KeyframeExtractor:
    """Handles extraction of keyframes from video files."""

    def __init__(self, frame_interval: int = 60, quality: int = 85):
        """
        Initialize the KeyframeExtractor.

        Args:
            frame_interval: Extract a frame every N seconds (default: 60)
            quality: JPEG quality for saved frames (1-100, default: 85)
        """
        self.frame_interval = frame_interval
        self.quality = quality

    def extract_frame_at_timestamp(
        self,
        video_path: Union[str, Path],
        timestamp: float,
        output_path: Union[str, Path]
    ) -> Path:
        """
        Extract a single frame at a specific timestamp.

        Args:
            video_path: Path to the video file
            timestamp: Time in seconds
            output_path: Path to save the frame

        Returns:
            Path to the saved frame

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If timestamp is invalid
        """
        video_path = Path(video_path)
        output_path = Path(output_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open video
        video = VideoFileClip(str(video_path))

        if timestamp < 0 or timestamp > video.duration:
            video.close()
            raise ValueError(
                f"Timestamp {timestamp} out of range (0-{video.duration})"
            )

        # Extract frame at timestamp
        frame = video.get_frame(timestamp)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Save frame
        cv2.imwrite(
            str(output_path),
            frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )

        video.close()

        return output_path

    def extract_frames_for_timerange(
        self,
        video_path: Union[str, Path],
        start_time: float,
        end_time: float,
        output_dir: Union[str, Path],
        max_frames: int = 6
    ) -> List[Tuple[Path, float]]:
        """
        Extract evenly spaced frames within a time range.

        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_dir: Directory to save frames
            max_frames: Maximum number of frames to extract

        Returns:
            List of tuples (frame_path, timestamp)

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If time range is invalid
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if start_time >= end_time:
            raise ValueError(f"Invalid time range: {start_time} >= {end_time}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        duration = end_time - start_time

        # Calculate number of frames to extract
        # Use either max_frames or one frame per interval, whichever is smaller
        num_frames_by_interval = int(duration / self.frame_interval) + 1
        num_frames = min(max_frames, num_frames_by_interval)

        # Ensure at least 1 frame
        num_frames = max(1, num_frames)

        # Calculate timestamps
        if num_frames == 1:
            # Extract frame from the middle
            timestamps = [start_time + duration / 2]
        else:
            # Evenly space frames
            interval = duration / (num_frames - 1) if num_frames > 1 else 0
            timestamps = [start_time + i * interval for i in range(num_frames)]

        # Extract frames
        frames = []
        video = VideoFileClip(str(video_path))

        print(f"Extracting {num_frames} keyframes from {start_time:.2f}s to {end_time:.2f}s")

        for i, timestamp in enumerate(timestamps):
            try:
                # Generate output filename
                frame_path = output_dir / f"frame_{i + 1:03d}.jpg"

                # Extract frame
                frame = video.get_frame(timestamp)

                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Save frame
                cv2.imwrite(
                    str(frame_path),
                    frame_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )

                frames.append((frame_path, timestamp))

            except Exception as e:
                print(f"Warning: Failed to extract frame at {timestamp:.2f}s: {e}")
                continue

        video.close()

        print(f"Successfully extracted {len(frames)} keyframes")

        return frames

    def extract_frames_with_scene_change(
        self,
        video_path: Union[str, Path],
        start_time: float,
        end_time: float,
        output_dir: Union[str, Path],
        max_frames: int = 6,
        threshold: float = 30.0
    ) -> List[Tuple[Path, float]]:
        """
        Extract frames at scene changes within a time range.

        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_dir: Directory to save frames
            max_frames: Maximum number of frames to extract
            threshold: Scene change detection threshold (higher = more sensitive)

        Returns:
            List of tuples (frame_path, timestamp)
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            cap.release()
            # Fallback to evenly spaced frames
            return self.extract_frames_for_timerange(
                video_path, start_time, end_time, output_dir, max_frames
            )

        # Convert to frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Set to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        scene_changes = []
        prev_frame = None
        current_frame_num = start_frame

        print(f"Detecting scene changes from {start_time:.2f}s to {end_time:.2f}s")

        while current_frame_num < end_frame:
            ret, frame = cap.read()

            if not ret:
                break

            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Calculate difference
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff)

                if mean_diff > threshold:
                    timestamp = current_frame_num / fps
                    scene_changes.append((current_frame_num, timestamp, frame.copy()))

            prev_frame = gray
            current_frame_num += 1

        cap.release()

        # If we found fewer scene changes than max_frames, add evenly spaced frames
        if len(scene_changes) < max_frames:
            print(f"Found {len(scene_changes)} scene changes, filling with evenly spaced frames")
            evenly_spaced = self.extract_frames_for_timerange(
                video_path, start_time, end_time, output_dir, max_frames - len(scene_changes)
            )

            # Save scene change frames
            frames = []
            for i, (frame_num, timestamp, frame_data) in enumerate(scene_changes):
                frame_path = output_dir / f"frame_{i + 1:03d}_scene.jpg"
                cv2.imwrite(
                    str(frame_path),
                    frame_data,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                frames.append((frame_path, timestamp))

            # Add evenly spaced frames
            frames.extend(evenly_spaced)
            return frames

        # Sort by timestamp and limit to max_frames
        scene_changes.sort(key=lambda x: x[1])
        scene_changes = scene_changes[:max_frames]

        # Save frames
        frames = []
        for i, (frame_num, timestamp, frame_data) in enumerate(scene_changes):
            frame_path = output_dir / f"frame_{i + 1:03d}.jpg"
            cv2.imwrite(
                str(frame_path),
                frame_data,
                [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            )
            frames.append((frame_path, timestamp))

        print(f"Extracted {len(frames)} keyframes at scene changes")

        return frames
