"""Module for processing individual chapters with VLM."""

from pathlib import Path
from typing import Union
from tqdm import tqdm
from .llm import VLMClient, VLMResponseParseError
from .models import Chapter, ChapterResult
from .keyframe_extractor import KeyframeExtractor
from .utils import format_timestamp


class ChapterProcessor:
    """Processes individual chapters with VLM for correction and summarization."""

    def __init__(
        self,
        vlm_client: VLMClient,
        keyframe_extractor: KeyframeExtractor
    ):
        """
        Initialize the ChapterProcessor.

        Args:
            vlm_client: VLM client instance
            keyframe_extractor: KeyframeExtractor instance
        """
        self.vlm_client = vlm_client
        self.keyframe_extractor = keyframe_extractor

    def _build_correction_prompt(self, chapter: Chapter) -> str:
        """
        Build the prompt for chapter correction and summarization.

        Args:
            chapter: Chapter object

        Returns:
            Formatted prompt string
        """
        # Get transcript text
        transcript_lines = []
        for segment in chapter.transcript_segments:
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            transcript_lines.append(f"[{start:.2f} - {end:.2f}] {text}")

        transcript_text = '\n'.join(transcript_lines)

        prompt = f"""You are reviewing Chapter {chapter.index} of a video: "{chapter.title}"

**Chapter Information:**
- Time range: {format_timestamp(chapter.start_time)} - {format_timestamp(chapter.end_time)}
- Duration: {chapter.duration:.1f} seconds ({chapter.duration/60:.1f} minutes)

**Raw Transcript (from speech-to-text, may contain errors):**
{transcript_text}

**Images:**
I'm providing key frames from this chapter to help you understand the visual context.

**Your Tasks:**
1. **Correct the transcript**: Review the raw transcript and correct any obvious speech-to-text errors, typos, or misheard words. Use the visual context from the keyframes to help identify corrections.

2. **Summarize the chapter**: Write a comprehensive summary (3-5 sentences) that captures the main content and key takeaways from this chapter.

3. **Extract key points**: Identify 3-5 key points or main ideas covered in this chapter.

**Important Guidelines:**
- Keep the corrected transcript in the same timestamped format
- Make corrections only where you're confident there's an error
- The summary should be informative and self-contained
- Key points should be concise bullet points

**Output Format:**
Return a valid JSON object with this exact structure:
{{
  "corrected_transcript": "[0.00 - 5.23] Corrected text here\\n[5.23 - 12.45] More corrected text...",
  "summary": "This chapter covers...",
  "key_points": [
    "First key point",
    "Second key point",
    "Third key point"
  ]
}}

IMPORTANT: Respond ONLY with the JSON object, no additional text or explanation."""

        return prompt

    def process_chapter(
        self,
        chapter: Chapter,
        video_path: Union[str, Path],
        output_dir: Union[str, Path]
    ) -> ChapterResult:
        """
        Process a chapter: extract keyframes and use VLM for correction/summary.

        Args:
            chapter: Chapter object
            video_path: Path to the video file
            output_dir: Directory for output files

        Returns:
            ChapterResult object
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        # Create chapter-specific directory
        chapter_dir = output_dir / f"chapter_{chapter.index:03d}"
        chapter_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing Chapter {chapter.index}: {chapter.title}")
        print(f"Time: {format_timestamp(chapter.start_time)} - {format_timestamp(chapter.end_time)}")
        print(f"{'='*60}\n")

        # Step 1: Extract keyframes
        print("Extracting keyframes...")
        try:
            keyframes = self.keyframe_extractor.extract_frames_for_timerange(
                video_path,
                chapter.start_time,
                chapter.end_time,
                chapter_dir,
                max_frames=6
            )
            keyframe_paths = [kf[0] for kf in keyframes]
            print(f"Extracted {len(keyframe_paths)} keyframes")
        except Exception as e:
            print(f"Error extracting keyframes: {e}")
            keyframe_paths = []

        # Step 2: Build prompt
        prompt = self._build_correction_prompt(chapter)

        # Step 3: Send to VLM
        print("Sending to VLM for correction and summarization...")
        try:
            response = self.vlm_client.send_message_with_json(
                prompt,
                images=keyframe_paths if keyframe_paths else None
            )

            # Parse response
            corrected_transcript = response.get('corrected_transcript', chapter.get_transcript_text())
            summary = response.get('summary', '')
            key_points = response.get('key_points', [])

            if not summary:
                print("Warning: VLM returned empty summary")
                summary = "Summary not available."

            if not key_points:
                print("Warning: VLM returned no key points")
                key_points = ["Key points not available."]

            result = ChapterResult(
                chapter=chapter,
                corrected_transcript=corrected_transcript,
                summary=summary,
                key_points=key_points,
                keyframes=keyframe_paths,
                error=None
            )

            print(f"✓ Successfully processed chapter")

        except VLMResponseParseError as e:
            print(f"Error parsing VLM response: {e}")
            result = self._fallback_processing(chapter, keyframe_paths, str(e))

        except Exception as e:
            print(f"Error processing chapter: {e}")
            result = self._fallback_processing(chapter, keyframe_paths, str(e))

        return result

    def _fallback_processing(
        self,
        chapter: Chapter,
        keyframe_paths: list,
        error_message: str
    ) -> ChapterResult:
        """
        Fallback processing when VLM fails.

        Args:
            chapter: Chapter object
            keyframe_paths: List of keyframe paths
            error_message: Error message

        Returns:
            ChapterResult with basic information
        """
        print("Using fallback: returning raw transcript without correction")

        # Get raw transcript
        transcript_lines = []
        for segment in chapter.transcript_segments:
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            transcript_lines.append(f"[{start:.2f} - {end:.2f}] {text}")

        corrected_transcript = '\n'.join(transcript_lines)

        # Generate basic summary
        summary = f"Chapter {chapter.index}: {chapter.title}. Duration: {chapter.duration:.1f} seconds."

        return ChapterResult(
            chapter=chapter,
            corrected_transcript=corrected_transcript,
            summary=summary,
            key_points=["VLM processing failed - raw transcript provided"],
            keyframes=keyframe_paths,
            error=error_message
        )

    def process_all_chapters(
        self,
        chapters: list,
        video_path: Union[str, Path],
        output_dir: Union[str, Path],
        checkpoint_path: Union[str, Path] = None
    ) -> list:
        """
        Process all chapters with progress tracking and checkpointing.

        Args:
            chapters: List of Chapter objects
            video_path: Path to video file
            output_dir: Directory for output files
            checkpoint_path: Path to save checkpoints (optional)

        Returns:
            List of ChapterResult objects
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)

        # Load checkpoint if exists
        results = []
        start_index = 0

        if checkpoint_path and checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            import json
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                results = [ChapterResult.from_dict(d) for d in checkpoint_data]
                start_index = len(results)
            print(f"Resuming from chapter {start_index + 1}")

        # Process remaining chapters
        for i in tqdm(range(start_index, len(chapters)), desc="Processing chapters"):
            chapter = chapters[i]

            result = self.process_chapter(chapter, video_path, output_dir)
            results.append(result)

            # Save checkpoint
            if checkpoint_path:
                import json
                with open(checkpoint_path, 'w') as f:
                    json.dump([r.to_dict() for r in results], f, indent=2)

        # Clean up checkpoint
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()

        return results
