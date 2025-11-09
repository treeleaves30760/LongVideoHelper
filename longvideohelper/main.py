"""Main CLI module for LongVideoHelper."""

import click
import os
from pathlib import Path
from dotenv import load_dotenv
from .audio_extractor import AudioExtractor
from .audio_clipper import AudioClipper
from .transcriber import Transcriber
from .llm import VLMClient
from .keyframe_extractor import KeyframeExtractor
from .chapter_detector import ChapterDetector
from .chapter_processor import ChapterProcessor
from .markdown_generator import MarkdownGenerator
from .models import VideoMetadata
from .utils import parse_transcript_file, get_video_duration

# Load environment variables
load_dotenv()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """LongVideoHelper - A tool for long video transcription and chapter division."""
    pass


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='output',
    help='Output directory for results'
)
@click.option(
    '--model',
    '-m',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'turbo']),
    default='turbo',
    help='Whisper model to use'
)
@click.option(
    '--language',
    '-l',
    type=str,
    default=None,
    help='Language code (e.g., en, zh) or auto-detect if not specified'
)
@click.option(
    '--max-clip-duration',
    type=int,
    default=300,
    help='Maximum duration of each clip in seconds (default: 300 = 5 minutes)'
)
@click.option(
    '--keep-clips',
    is_flag=True,
    help='Keep intermediate audio clips (default: delete after transcription)'
)
def transcribe(video_path, output_dir, model, language, max_clip_duration, keep_clips):
    """
    Transcribe a video file using Whisper.

    This command extracts audio from the video, clips it using VAD,
    and transcribes each clip using Whisper.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n{'='*60}")
    click.echo(f"LongVideoHelper - Video Transcription")
    click.echo(f"{'='*60}\n")
    click.echo(f"Video: {video_path}")
    click.echo(f"Output Directory: {output_dir}")
    click.echo(f"Whisper Model: {model}")
    click.echo(f"Language: {language or 'auto-detect'}")
    click.echo(f"Max Clip Duration: {max_clip_duration}s\n")

    # Step 1: Extract audio
    click.echo("Step 1/4: Extracting audio from video...")
    extractor = AudioExtractor()
    audio_path = output_dir / f"{video_path.stem}_audio.wav"
    try:
        audio_path = extractor.extract_audio(video_path, audio_path)
    except Exception as e:
        click.echo(f"Error extracting audio: {str(e)}", err=True)
        return

    # Step 2: Clip audio using VAD
    click.echo("\nStep 2/4: Clipping audio using Voice Activity Detection...")
    clipper = AudioClipper(max_clip_duration=max_clip_duration)
    clips_dir = output_dir / "clips"

    try:
        clips = clipper.clip_audio(audio_path, clips_dir)
    except Exception as e:
        click.echo(f"Error clipping audio: {str(e)}", err=True)
        return

    if not clips:
        click.echo("No audio clips generated. The video may not contain speech.", err=True)
        return

    click.echo(f"Generated {len(clips)} audio clips")

    # Step 3: Transcribe clips
    click.echo("\nStep 3/4: Transcribing audio clips...")
    transcriber = Transcriber(model_name=model)

    try:
        transcriptions = transcriber.transcribe_clips(clips, language=language)
    except Exception as e:
        click.echo(f"Error transcribing audio: {str(e)}", err=True)
        return

    # Step 4: Merge and save results
    click.echo("\nStep 4/4: Saving transcription results...")
    merged = transcriber.merge_transcriptions(transcriptions)

    # Save full transcription with timestamps
    transcript_path = output_dir / f"{video_path.stem}_transcript.txt"
    transcriber.save_transcription(merged, transcript_path, include_timestamps=True)

    # Save plain text version
    plain_text_path = output_dir / f"{video_path.stem}_transcript_plain.txt"
    transcriber.save_transcription(merged, plain_text_path, include_timestamps=False)

    # Clean up intermediate files if requested
    if not keep_clips:
        click.echo("\nCleaning up intermediate files...")
        for clip_path, _, _ in clips:
            clip_path.unlink(missing_ok=True)
        clips_dir.rmdir()
        audio_path.unlink(missing_ok=True)

    # Print summary
    click.echo(f"\n{'='*60}")
    click.echo("Transcription Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"\nResults saved to:")
    click.echo(f"  - Transcript (with timestamps): {transcript_path}")
    click.echo(f"  - Transcript (plain text): {plain_text_path}")
    click.echo(f"\nDetected language: {merged['language']}")
    click.echo(f"Total segments: {len(merged['segments'])}")
    click.echo(f"Total clips processed: {len(clips)}\n")


@cli.command()
@click.argument('audio_path', type=click.Path(exists=True))
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='output',
    help='Output directory for results'
)
@click.option(
    '--model',
    '-m',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'turbo']),
    default='turbo',
    help='Whisper model to use'
)
@click.option(
    '--language',
    '-l',
    type=str,
    default=None,
    help='Language code (e.g., en, zh) or auto-detect if not specified'
)
def transcribe_audio(audio_path, output_dir, model, language):
    """
    Transcribe an audio file directly (without clipping).

    This command transcribes an audio file using Whisper without
    performing VAD-based clipping.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n{'='*60}")
    click.echo(f"LongVideoHelper - Audio Transcription")
    click.echo(f"{'='*60}\n")
    click.echo(f"Audio: {audio_path}")
    click.echo(f"Output Directory: {output_dir}")
    click.echo(f"Whisper Model: {model}")
    click.echo(f"Language: {language or 'auto-detect'}\n")

    click.echo("Transcribing audio...")
    transcriber = Transcriber(model_name=model)

    try:
        result = transcriber.transcribe_clip(audio_path, language=language)
    except Exception as e:
        click.echo(f"Error transcribing audio: {str(e)}", err=True)
        return

    # Save results
    transcript_path = output_dir / f"{audio_path.stem}_transcript.txt"
    transcriber.save_transcription(result, transcript_path, include_timestamps=True)

    plain_text_path = output_dir / f"{audio_path.stem}_transcript_plain.txt"
    transcriber.save_transcription(result, plain_text_path, include_timestamps=False)

    # Print summary
    click.echo(f"\n{'='*60}")
    click.echo("Transcription Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"\nResults saved to:")
    click.echo(f"  - Transcript (with timestamps): {transcript_path}")
    click.echo(f"  - Transcript (plain text): {plain_text_path}")
    click.echo(f"\nDetected language: {result.get('language', 'unknown')}")
    click.echo(f"Total segments: {len(result.get('segments', []))}\n")


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option(
    '--transcript',
    '-t',
    type=click.Path(exists=True),
    help='Path to existing transcript file (if not provided, will look in output-dir)'
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='output',
    help='Output directory for results'
)
@click.option(
    '--vlm-provider',
    type=click.Choice(['gemini', 'ollama', 'openai']),
    required=True,
    help='VLM provider (gemini, ollama, openai)'
)
@click.option(
    '--vlm-model',
    required=True,
    help='VLM model name (e.g., gemini/gemini-2.0-flash-exp, ollama/qwen2-vl:8b)'
)
@click.option(
    '--api-key',
    envvar='GEMINI_API_KEY',
    help='API key for cloud providers (or set GEMINI_API_KEY env var)'
)
@click.option(
    '--chapter-duration',
    type=int,
    default=360,
    help='Target chapter duration in seconds (default: 360 = 6 minutes)'
)
@click.option(
    '--max-keyframes',
    type=int,
    default=6,
    help='Maximum keyframes per chapter (default: 6)'
)
def create_chapters(video_path, transcript, output_dir, vlm_provider, vlm_model, api_key, chapter_duration, max_keyframes):
    """
    Create chapter summaries from an existing transcript using VLM.

    This command takes a video and its transcript, divides it into chapters,
    and generates summaries using a Vision-Language Model.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize model name (remove provider prefix if present)
    if vlm_model.startswith(f"{vlm_provider}/"):
        vlm_model_normalized = vlm_model
    else:
        vlm_model_normalized = f"{vlm_provider}/{vlm_model}"

    click.echo(f"\n{'='*60}")
    click.echo(f"LongVideoHelper - Chapter Creation")
    click.echo(f"{'='*60}\n")
    click.echo(f"Video: {video_path}")
    click.echo(f"Output Directory: {output_dir}")
    click.echo(f"VLM: {vlm_model_normalized}")
    click.echo(f"Target Chapter Duration: {chapter_duration}s\n")

    # Step 1: Load transcript
    if transcript:
        transcript_path = Path(transcript)
    else:
        # Look for transcript in output directory
        transcript_path = output_dir / f"{video_path.stem}_transcript.txt"
        if not transcript_path.exists():
            click.echo(f"Error: Transcript not found at {transcript_path}", err=True)
            click.echo("Please specify transcript path with --transcript option or run 'transcribe' first.", err=True)
            return

    click.echo(f"Step 1/5: Loading transcript from {transcript_path}")
    try:
        transcript_data = parse_transcript_file(transcript_path)
    except Exception as e:
        click.echo(f"Error loading transcript: {str(e)}", err=True)
        return

    # Get video duration
    try:
        video_duration = get_video_duration(video_path)
    except Exception as e:
        click.echo(f"Error getting video duration: {str(e)}", err=True)
        return

    # Step 2: Initialize VLM client
    click.echo(f"\nStep 2/5: Initializing VLM client...")
    try:
        vlm_client = VLMClient(
            provider=vlm_provider,
            model_name=vlm_model_normalized,
            api_key=api_key
        )
    except Exception as e:
        click.echo(f"Error initializing VLM client: {str(e)}", err=True)
        return

    # Step 3: Detect chapters
    click.echo(f"\nStep 3/5: Detecting chapter boundaries...")
    detector = ChapterDetector(
        vlm_client=vlm_client,
        target_duration=chapter_duration
    )

    try:
        chapters = detector.detect_chapters(transcript_data, video_duration)
    except Exception as e:
        click.echo(f"Error detecting chapters: {str(e)}", err=True)
        return

    if not chapters:
        click.echo("No chapters detected.", err=True)
        return

    click.echo(f"Detected {len(chapters)} chapters")

    # Step 4: Process chapters
    click.echo(f"\nStep 4/5: Processing chapters with VLM...")
    keyframe_extractor = KeyframeExtractor(frame_interval=60)
    processor = ChapterProcessor(vlm_client, keyframe_extractor)

    chapters_dir = output_dir / "chapters"
    checkpoint_path = output_dir / "chapter_checkpoint.json"

    try:
        results = processor.process_all_chapters(
            chapters,
            video_path,
            chapters_dir,
            checkpoint_path
        )
    except Exception as e:
        click.echo(f"Error processing chapters: {str(e)}", err=True)
        return

    # Step 5: Generate outputs
    click.echo(f"\nStep 5/5: Generating output files...")
    generator = MarkdownGenerator()

    # Create video metadata
    metadata = VideoMetadata(
        path=video_path,
        duration=video_duration,
        language=transcript_data.get('language', 'unknown'),
        total_chapters=len(chapters)
    )

    # Generate full markdown summary
    markdown_path = output_dir / f"{video_path.stem}_chapters.md"
    generator.generate_chapter_summary(
        video_path.stem,
        results,
        markdown_path,
        metadata
    )

    # Generate JSON output
    json_path = output_dir / f"{video_path.stem}_chapters.json"
    generator.generate_json_output(
        video_path.stem,
        results,
        json_path,
        metadata
    )

    # Generate quick summary
    quick_summary_path = output_dir / f"{video_path.stem}_summary.md"
    generator.generate_simple_summary(
        video_path.stem,
        results,
        quick_summary_path
    )

    # Print summary
    click.echo(f"\n{'='*60}")
    click.echo("Chapter Creation Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"\nResults saved to:")
    click.echo(f"  - Full Summary: {markdown_path}")
    click.echo(f"  - Quick Summary: {quick_summary_path}")
    click.echo(f"  - JSON Data: {json_path}")
    click.echo(f"  - Keyframes: {chapters_dir}/")
    click.echo(f"\nTotal chapters: {len(results)}")
    successful = sum(1 for r in results if r.success)
    click.echo(f"Successfully processed: {successful}/{len(results)}\n")


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='output',
    help='Output directory for results'
)
@click.option(
    '--whisper-model',
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'turbo']),
    default='turbo',
    help='Whisper model to use for transcription'
)
@click.option(
    '--vlm-provider',
    type=click.Choice(['gemini', 'ollama', 'openai']),
    required=True,
    help='VLM provider for chapter processing'
)
@click.option(
    '--vlm-model',
    required=True,
    help='VLM model name'
)
@click.option(
    '--api-key',
    envvar='GEMINI_API_KEY',
    help='API key for cloud providers'
)
@click.option(
    '--language',
    '-l',
    type=str,
    default=None,
    help='Language code for transcription'
)
@click.option(
    '--chapter-duration',
    type=int,
    default=360,
    help='Target chapter duration in seconds'
)
def process(video_path, output_dir, whisper_model, vlm_provider, vlm_model, api_key, language, chapter_duration):
    """
    Full pipeline: transcribe video and create chapter summaries.

    This command runs both Phase 1 (transcription) and Phase 2 (chapter creation)
    in a single workflow.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize model name (remove provider prefix if present)
    if vlm_model.startswith(f"{vlm_provider}/"):
        vlm_model_normalized = vlm_model
    else:
        vlm_model_normalized = f"{vlm_provider}/{vlm_model}"

    click.echo(f"\n{'='*60}")
    click.echo(f"LongVideoHelper - Full Processing Pipeline")
    click.echo(f"{'='*60}\n")
    click.echo(f"Video: {video_path}")
    click.echo(f"Output Directory: {output_dir}")
    click.echo(f"Whisper Model: {whisper_model}")
    click.echo(f"VLM: {vlm_model_normalized}\n")

    # PHASE 1: Transcription
    click.echo(f"\n{'#'*60}")
    click.echo("PHASE 1: TRANSCRIPTION")
    click.echo(f"{'#'*60}\n")

    # Extract audio
    click.echo("Step 1/4: Extracting audio from video...")
    extractor = AudioExtractor()
    audio_path = output_dir / f"{video_path.stem}_audio.wav"
    try:
        audio_path = extractor.extract_audio(video_path, audio_path)
    except Exception as e:
        click.echo(f"Error extracting audio: {str(e)}", err=True)
        return

    # Clip audio
    click.echo("\nStep 2/4: Clipping audio using VAD...")
    clipper = AudioClipper(max_clip_duration=300)
    clips_dir = output_dir / "clips"

    try:
        clips = clipper.clip_audio(audio_path, clips_dir)
    except Exception as e:
        click.echo(f"Error clipping audio: {str(e)}", err=True)
        return

    # Transcribe
    click.echo("\nStep 3/4: Transcribing audio clips...")
    transcriber = Transcriber(model_name=whisper_model)

    try:
        transcriptions = transcriber.transcribe_clips(clips, language=language)
    except Exception as e:
        click.echo(f"Error transcribing audio: {str(e)}", err=True)
        return

    # Save transcript
    click.echo("\nStep 4/4: Saving transcription...")
    merged = transcriber.merge_transcriptions(transcriptions)
    transcript_path = output_dir / f"{video_path.stem}_transcript.txt"
    transcriber.save_transcription(merged, transcript_path, include_timestamps=True)

    # Clean up
    for clip_path, _, _ in clips:
        clip_path.unlink(missing_ok=True)
    clips_dir.rmdir()
    audio_path.unlink(missing_ok=True)

    click.echo(f"✓ Transcription complete")

    # PHASE 2: Chapter Creation
    click.echo(f"\n{'#'*60}")
    click.echo("PHASE 2: CHAPTER CREATION")
    click.echo(f"{'#'*60}\n")

    # Get video duration
    video_duration = get_video_duration(video_path)

    # Initialize VLM
    click.echo("Step 1/4: Initializing VLM client...")
    vlm_client = VLMClient(provider=vlm_provider, model_name=vlm_model_normalized, api_key=api_key)

    # Detect chapters
    click.echo("\nStep 2/4: Detecting chapter boundaries...")
    detector = ChapterDetector(vlm_client, target_duration=chapter_duration)
    chapters = detector.detect_chapters({'text': merged['text'], 'segments': merged['segments']}, video_duration)

    # Process chapters
    click.echo("\nStep 3/4: Processing chapters...")
    keyframe_extractor = KeyframeExtractor()
    processor = ChapterProcessor(vlm_client, keyframe_extractor)
    results = processor.process_all_chapters(chapters, video_path, output_dir / "chapters")

    # Generate outputs
    click.echo("\nStep 4/4: Generating outputs...")
    generator = MarkdownGenerator()
    metadata = VideoMetadata(
        path=video_path,
        duration=video_duration,
        language=merged['language'],
        total_chapters=len(chapters)
    )

    markdown_path = output_dir / f"{video_path.stem}_chapters.md"
    generator.generate_chapter_summary(video_path.stem, results, markdown_path, metadata)

    json_path = output_dir / f"{video_path.stem}_chapters.json"
    generator.generate_json_output(video_path.stem, results, json_path, metadata)

    # Print summary
    click.echo(f"\n{'='*60}")
    click.echo("FULL PROCESSING COMPLETE!")
    click.echo(f"{'='*60}")
    click.echo(f"\nAll results saved to: {output_dir}/")
    click.echo(f"  - Transcript: {transcript_path}")
    click.echo(f"  - Chapter Summary: {markdown_path}")
    click.echo(f"  - JSON Data: {json_path}")
    click.echo(f"\nTotal chapters: {len(results)}\n")


if __name__ == '__main__':
    cli()
