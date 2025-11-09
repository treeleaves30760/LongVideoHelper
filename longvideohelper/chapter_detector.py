"""Module for detecting chapter boundaries using VLM."""

from typing import List, Dict
from .llm import VLMClient, VLMResponseParseError
from .models import Chapter
from .utils import get_segments_in_timerange


class ChapterDetector:
    """Detects chapter boundaries in a video transcript using VLM."""

    def __init__(
        self,
        vlm_client: VLMClient,
        target_duration: int = 360,
        min_duration: int = 180,
        max_duration: int = 600
    ):
        """
        Initialize the ChapterDetector.

        Args:
            vlm_client: VLM client instance
            target_duration: Target chapter duration in seconds (default: 360 = 6 minutes)
            min_duration: Minimum chapter duration in seconds (default: 180 = 3 minutes)
            max_duration: Maximum chapter duration in seconds (default: 600 = 10 minutes)
        """
        self.vlm_client = vlm_client
        self.target_duration = target_duration
        self.min_duration = min_duration
        self.max_duration = max_duration

    def _build_detection_prompt(
        self,
        transcript_text: str,
        duration: float,
        num_suggested_chapters: int
    ) -> str:
        """
        Build the prompt for chapter detection.

        Args:
            transcript_text: Full transcript text
            duration: Video duration in seconds
            num_suggested_chapters: Suggested number of chapters

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are analyzing a video transcript to divide it into logical chapters.

**Video Information:**
- Total duration: {duration:.2f} seconds ({duration/60:.1f} minutes)
- Suggested number of chapters: {num_suggested_chapters}
- Target chapter duration: ~{self.target_duration} seconds ({self.target_duration/60:.1f} minutes)
- Acceptable range: {self.min_duration}-{self.max_duration} seconds

**Transcript:**
{transcript_text}

**Task:**
Divide this transcript into logical chapters based on topic boundaries and natural breaks in the content.
Each chapter should be approximately {self.target_duration} seconds long, but can vary based on natural topic boundaries.

**Requirements:**
1. Create {num_suggested_chapters} chapters (you can create fewer or more if it makes sense)
2. Each chapter should cover a distinct topic or theme
3. Chapters should not be too short (minimum {self.min_duration}s) or too long (maximum {self.max_duration}s)
4. Give each chapter a descriptive title
5. Ensure all chapters together cover the entire video (0 to {duration:.2f} seconds)

**Output Format:**
Return a valid JSON object with this exact structure:
{{
  "chapters": [
    {{
      "index": 1,
      "start_time": 0.0,
      "end_time": 360.0,
      "title": "Introduction and Setup"
    }},
    {{
      "index": 2,
      "start_time": 360.0,
      "end_time": 720.0,
      "title": "Main Concepts"
    }}
  ]
}}

IMPORTANT: Respond ONLY with the JSON object, no additional text or explanation."""

        return prompt

    def _parse_chapter_response(
        self,
        response: Dict,
        all_segments: List[Dict]
    ) -> List[Chapter]:
        """
        Parse VLM response into Chapter objects.

        Args:
            response: Parsed JSON response from VLM
            all_segments: All transcript segments

        Returns:
            List of Chapter objects

        Raises:
            VLMResponseParseError: If response format is invalid
        """
        if 'chapters' not in response:
            raise VLMResponseParseError("Response missing 'chapters' key")

        chapters = []

        for chapter_data in response['chapters']:
            # Validate required fields
            required_fields = ['index', 'start_time', 'end_time', 'title']
            for field in required_fields:
                if field not in chapter_data:
                    raise VLMResponseParseError(f"Chapter missing required field: {field}")

            # Extract chapter data
            index = chapter_data['index']
            start_time = float(chapter_data['start_time'])
            end_time = float(chapter_data['end_time'])
            title = chapter_data['title']

            # Get transcript segments for this chapter
            chapter_segments = get_segments_in_timerange(
                all_segments,
                start_time,
                end_time
            )

            # Create Chapter object
            chapter = Chapter(
                index=index,
                start_time=start_time,
                end_time=end_time,
                title=title,
                transcript_segments=chapter_segments
            )

            chapters.append(chapter)

        return chapters

    def _validate_chapters(
        self,
        chapters: List[Chapter],
        video_duration: float
    ) -> List[Chapter]:
        """
        Validate and fix chapter boundaries.

        Args:
            chapters: List of Chapter objects
            video_duration: Total video duration

        Returns:
            Validated and fixed list of chapters
        """
        if not chapters:
            print("Warning: No chapters detected")
            return []

        # Sort by start time
        chapters.sort(key=lambda c: c.start_time)

        validated = []

        for i, chapter in enumerate(chapters):
            # Check minimum duration
            if chapter.duration < self.min_duration:
                print(f"Warning: Chapter {chapter.index} is too short ({chapter.duration:.1f}s)")

                # Try to merge with next chapter
                if i + 1 < len(chapters):
                    print(f"  Merging with Chapter {chapters[i+1].index}")
                    continue

            # Check maximum duration
            if chapter.duration > self.max_duration:
                print(f"Warning: Chapter {chapter.index} is too long ({chapter.duration:.1f}s)")
                # Still keep it, but warn user

            # Check for gaps
            if validated and chapter.start_time > validated[-1].end_time + 1:
                print(f"Warning: Gap detected between Chapter {validated[-1].index} and {chapter.index}")

            # Check for overlaps
            if validated and chapter.start_time < validated[-1].end_time:
                print(f"Warning: Overlap detected between Chapter {validated[-1].index} and {chapter.index}")
                # Adjust start time
                chapter.start_time = validated[-1].end_time

            validated.append(chapter)

        # Ensure first chapter starts at 0
        if validated and validated[0].start_time > 0:
            print(f"Adjusting first chapter to start at 0")
            validated[0].start_time = 0.0

        # Ensure last chapter ends at video duration
        if validated and validated[-1].end_time < video_duration:
            print(f"Adjusting last chapter to end at video duration")
            validated[-1].end_time = video_duration

        return validated

    def detect_chapters(
        self,
        transcript: Dict,
        video_duration: float
    ) -> List[Chapter]:
        """
        Detect chapter boundaries in the transcript.

        Args:
            transcript: Transcript dictionary with 'segments' and 'text'
            video_duration: Total video duration in seconds

        Returns:
            List of Chapter objects

        Raises:
            VLMResponseParseError: If VLM response cannot be parsed
        """
        transcript_text = transcript.get('text', '')
        all_segments = transcript.get('segments', [])

        # Calculate suggested number of chapters
        num_suggested_chapters = max(1, int(video_duration / self.target_duration))

        print(f"\nDetecting chapters...")
        print(f"Video duration: {video_duration:.1f}s ({video_duration/60:.1f} minutes)")
        print(f"Target chapter duration: {self.target_duration}s ({self.target_duration/60:.1f} minutes)")
        print(f"Suggested number of chapters: {num_suggested_chapters}")

        # Build prompt
        prompt = self._build_detection_prompt(
            transcript_text,
            video_duration,
            num_suggested_chapters
        )

        # Get VLM response
        print("Sending transcript to VLM for chapter detection...")
        try:
            response = self.vlm_client.send_message_with_json(prompt)
        except VLMResponseParseError as e:
            print(f"Error parsing VLM response: {e}")
            print("Falling back to automatic chapter division...")
            return self._fallback_chapter_detection(video_duration, all_segments)

        # Parse response
        try:
            chapters = self._parse_chapter_response(response, all_segments)
        except VLMResponseParseError as e:
            print(f"Error parsing chapter data: {e}")
            print("Falling back to automatic chapter division...")
            return self._fallback_chapter_detection(video_duration, all_segments)

        # Validate chapters
        chapters = self._validate_chapters(chapters, video_duration)

        print(f"Successfully detected {len(chapters)} chapters")

        return chapters

    def _fallback_chapter_detection(
        self,
        video_duration: float,
        all_segments: List[Dict]
    ) -> List[Chapter]:
        """
        Fallback method: divide video into equal-duration chapters.

        Args:
            video_duration: Total video duration
            all_segments: All transcript segments

        Returns:
            List of Chapter objects
        """
        print("Using fallback: dividing video into equal-duration chapters")

        num_chapters = max(1, int(video_duration / self.target_duration))
        chapter_duration = video_duration / num_chapters

        chapters = []

        for i in range(num_chapters):
            start_time = i * chapter_duration
            end_time = min((i + 1) * chapter_duration, video_duration)

            # Get segments for this chapter
            chapter_segments = get_segments_in_timerange(
                all_segments,
                start_time,
                end_time
            )

            chapter = Chapter(
                index=i + 1,
                start_time=start_time,
                end_time=end_time,
                title=f"Chapter {i + 1}",
                transcript_segments=chapter_segments
            )

            chapters.append(chapter)

        return chapters
