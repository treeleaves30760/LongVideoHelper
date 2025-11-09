"""Module for generating markdown documentation from chapter results."""

from pathlib import Path
from typing import List, Union
from datetime import datetime
from .models import ChapterResult, VideoMetadata
from .utils import format_timestamp, format_duration


class MarkdownGenerator:
    """Generates markdown documentation for video chapters."""

    def __init__(self):
        """Initialize the MarkdownGenerator."""
        pass

    def _format_chapter_section(self, result: ChapterResult, base_path: Path) -> str:
        """
        Format a single chapter section in markdown.

        Args:
            result: ChapterResult object
            base_path: Base path for relative image paths

        Returns:
            Formatted markdown string
        """
        chapter = result.chapter

        # Build markdown
        md = []

        # Chapter header
        md.append(f"## Chapter {chapter.index}: {chapter.title}")
        md.append("")
        md.append(f"**Time:** {format_timestamp(chapter.start_time)} - {format_timestamp(chapter.end_time)} "
                  f"({format_duration(chapter.duration)})")
        md.append("")

        # Summary section
        md.append("### Summary")
        md.append("")
        md.append(result.summary)
        md.append("")

        # Key points section
        if result.key_points:
            md.append("### Key Points")
            md.append("")
            for point in result.key_points:
                md.append(f"- {point}")
            md.append("")

        # Corrected transcript section
        if result.corrected_transcript:
            md.append("### Transcript")
            md.append("")
            md.append("```")
            md.append(result.corrected_transcript)
            md.append("```")
            md.append("")

        # Keyframes section
        if result.keyframes:
            md.append("### Key Frames")
            md.append("")

            for i, keyframe in enumerate(result.keyframes):
                # Make path relative to base_path
                try:
                    relative_path = keyframe.relative_to(base_path)
                except ValueError:
                    # If paths are on different drives, use absolute path
                    relative_path = keyframe

                md.append(f"![Frame {i+1}]({relative_path})")
                md.append("")

        # Error note (if any)
        if result.error:
            md.append("> **Note:** This chapter encountered processing errors:")
            md.append(f"> {result.error}")
            md.append("")

        md.append("---")
        md.append("")

        return '\n'.join(md)

    def _generate_table_of_contents(self, results: List[ChapterResult]) -> str:
        """
        Generate table of contents.

        Args:
            results: List of ChapterResult objects

        Returns:
            Formatted markdown string
        """
        md = []
        md.append("## Table of Contents")
        md.append("")

        for result in results:
            chapter = result.chapter
            # Create anchor link
            anchor = f"chapter-{chapter.index}-{chapter.title.lower().replace(' ', '-')}"
            # Remove special characters from anchor
            import re
            anchor = re.sub(r'[^a-z0-9-]', '', anchor)

            time_str = f"{format_timestamp(chapter.start_time)} - {format_timestamp(chapter.end_time)}"
            md.append(f"{chapter.index}. [{chapter.title}](#{anchor}) ({time_str})")

        md.append("")
        md.append("---")
        md.append("")

        return '\n'.join(md)

    def generate_chapter_summary(
        self,
        video_name: str,
        results: List[ChapterResult],
        output_path: Union[str, Path],
        video_metadata: VideoMetadata = None
    ):
        """
        Generate complete markdown summary document.

        Args:
            video_name: Name of the video
            results: List of ChapterResult objects
            output_path: Path to save the markdown file
            video_metadata: Optional video metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        md = []

        # Title
        md.append(f"# Video Summary: {video_name}")
        md.append("")

        # Metadata section
        if video_metadata:
            md.append(f"**Original File:** {video_metadata.path.name}")
            md.append(f"**Total Duration:** {format_timestamp(video_metadata.duration)} "
                      f"({format_duration(video_metadata.duration)})")
            md.append(f"**Language:** {video_metadata.language}")
            md.append(f"**Number of Chapters:** {len(results)}")
        else:
            md.append(f"**Number of Chapters:** {len(results)}")

        md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append("")
        md.append("---")
        md.append("")

        # Table of contents
        toc = self._generate_table_of_contents(results)
        md.append(toc)

        # Chapter sections
        base_path = output_path.parent

        for result in results:
            chapter_md = self._format_chapter_section(result, base_path)
            md.append(chapter_md)

        # Write to file
        content = '\n'.join(md)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Markdown summary saved to: {output_path}")

    def generate_json_output(
        self,
        video_name: str,
        results: List[ChapterResult],
        output_path: Union[str, Path],
        video_metadata: VideoMetadata = None
    ):
        """
        Generate JSON output for programmatic access.

        Args:
            video_name: Name of the video
            results: List of ChapterResult objects
            output_path: Path to save the JSON file
            video_metadata: Optional video metadata
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build JSON structure
        data = {
            "video_name": video_name,
            "generated_at": datetime.now().isoformat(),
            "chapters": [result.to_dict() for result in results]
        }

        if video_metadata:
            data["metadata"] = video_metadata.to_dict()

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"JSON output saved to: {output_path}")

    def generate_simple_summary(
        self,
        video_name: str,
        results: List[ChapterResult],
        output_path: Union[str, Path]
    ):
        """
        Generate a simple, concise summary (without full transcripts).

        Args:
            video_name: Name of the video
            results: List of ChapterResult objects
            output_path: Path to save the summary file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        md = []

        # Title
        md.append(f"# {video_name} - Quick Summary")
        md.append("")
        md.append(f"**Chapters:** {len(results)}")
        md.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d')}")
        md.append("")

        # Chapter summaries (no full transcripts)
        for result in results:
            chapter = result.chapter

            md.append(f"## {chapter.index}. {chapter.title}")
            md.append(f"**Time:** {format_timestamp(chapter.start_time)} - {format_timestamp(chapter.end_time)}")
            md.append("")
            md.append(result.summary)
            md.append("")

            if result.key_points:
                for point in result.key_points:
                    md.append(f"- {point}")
                md.append("")

        # Write to file
        content = '\n'.join(md)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Quick summary saved to: {output_path}")
