"""Main CLI module for LongVideoHelper."""

import click
import logging
import os
from datetime import datetime
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
from .chapter_segmenter import ChapterSegmenter
from .utils import parse_transcript_file, get_video_duration

# Load environment variables
load_dotenv()


def _load_config(config_path=None):
    """Load configuration from TOML file."""
    path = Path(config_path) if config_path else Path('longvideohelper.toml')
    if not path.exists():
        return {}
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            logging.getLogger(__name__).warning(
                f"Config file found ({path}) but tomllib/tomli not available. "
                "Requires Python 3.11+ or 'pip install tomli'."
            )
            return {}
    with open(path, 'rb') as f:
        config = tomllib.load(f)
    logging.getLogger(__name__).info(f"Loaded config from: {path}")
    return config


@click.group()
@click.version_option(version="0.1.0")
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose (DEBUG) logging')
@click.option('--config', type=click.Path(), default=None, help='Config file path (default: longvideohelper.toml in cwd)')
@click.pass_context
def cli(ctx, verbose, config):
    """LongVideoHelper - A tool for long video transcription and chapter division."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    ctx.ensure_object(dict)
    ctx.obj['config'] = _load_config(config)


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
@click.option(
    '--initial-prompt',
    type=str,
    default=None,
    help='Initial prompt for Whisper (domain vocabulary, proper nouns, style hints)'
)
@click.option(
    '--prompt-file',
    type=click.Path(exists=True),
    default=None,
    help='Load initial prompt from a text file'
)
@click.option(
    '--beam-size',
    type=int,
    default=5,
    help='Beam search width (default: 5)'
)
@click.option(
    '--compute-type',
    type=click.Choice(['default', 'float16', 'int8_float16', 'int8', 'float32']),
    default='default',
    help='CTranslate2 compute type (default: auto-select)'
)
@click.option(
    '--no-hallucination-filter',
    is_flag=True,
    help='Disable hallucination filtering'
)
@click.option(
    '--vocab-file',
    type=click.Path(exists=True),
    default=None,
    help='Vocabulary file with hotwords and known error examples for LLM correction'
)
@click.option(
    '--max-segment-duration',
    type=int,
    default=10,
    help='Max subtitle segment duration in seconds (default: 10)'
)
@click.option(
    '--correction-model',
    type=str,
    default=None,
    help='LLM model for vocab-based correction (e.g. gemini/gemini-2.0-flash, ollama/qwen3.5:27b, transformers/Qwen/Qwen3.5-27B). Requires --vocab-file'
)
@click.option(
    '--max-clips',
    type=int,
    default=None,
    help='Only transcribe the first N clips (for development testing)'
)
@click.option(
    '--chapters',
    is_flag=True,
    help='Enable chapter detection after transcription'
)
@click.option(
    '--chapter-model',
    type=str,
    default=None,
    help='LLM model for chapter detection (defaults to --correction-model)'
)
@click.option(
    '--chapter-duration',
    type=int,
    default=None,
    help='Target chapter duration in seconds (default: 300 = 5 minutes)'
)
@click.option(
    '--fps',
    type=int,
    default=None,
    help='Frame rate for timecode export (default: 24)'
)
@click.pass_context
def transcribe(ctx, video_path, output_dir, model, language, max_clip_duration, keep_clips,
               initial_prompt, prompt_file, beam_size, compute_type, no_hallucination_filter,
               vocab_file, max_segment_duration, correction_model, max_clips,
               chapters, chapter_model, chapter_duration, fps):
    """
    Transcribe a video file using Whisper.

    This command extracts audio from the video, clips it using VAD,
    and transcribes each clip using Whisper.
    """
    # Resolve config
    config = ctx.obj.get('config', {})
    vocab_file = vocab_file or config.get('correction', {}).get('vocab_file')
    correction_model = correction_model or config.get('correction', {}).get('model')
    chapter_model = chapter_model or config.get('chapters', {}).get('model')
    chapter_duration = chapter_duration or config.get('chapters', {}).get('duration', 300)
    chapters = chapters or config.get('chapters', {}).get('enabled', False)
    fps = fps or config.get('output', {}).get('fps', 24)

    video_path = Path(video_path)
    output_dir = Path(output_dir) / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n{'='*60}")
    click.echo(f"LongVideoHelper - Video Transcription")
    click.echo(f"{'='*60}\n")
    click.echo(f"Video: {video_path}")
    click.echo(f"Output Directory: {output_dir}")
    click.echo(f"Whisper Model: {model}")
    click.echo(f"Language: {language or 'auto-detect'}")
    click.echo(f"Max Clip Duration: {max_clip_duration}s")

    # Load prompt from file if specified
    if prompt_file:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            initial_prompt = f.read().strip()
    if initial_prompt:
        click.echo(f"Initial Prompt: {initial_prompt[:80]}{'...' if len(initial_prompt) > 80 else ''}")
    click.echo()

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

    if max_clips is not None and max_clips < len(clips):
        click.echo(f"  (limiting to first {max_clips} clips for testing)")
        clips = clips[:max_clips]

    # Step 3: Transcribe clips
    click.echo("\nStep 3/4: Transcribing audio clips...")
    transcriber = Transcriber(
        model_name=model,
        compute_type=compute_type,
        beam_size=beam_size,
        initial_prompt=initial_prompt,
        hallucination_filter=not no_hallucination_filter,
        vocab_file=vocab_file,
        max_segment_duration=max_segment_duration,
        correction_model=correction_model,
        video_path=str(video_path),
    )

    try:
        transcriptions = transcriber.transcribe_clips(clips, language=language)
    except Exception as e:
        click.echo(f"Error transcribing audio: {str(e)}", err=True)
        return

    # Step 4: Merge and save results
    click.echo("\nStep 4/4: Saving transcription results...")
    merged = transcriber.merge_transcriptions(transcriptions)

    # Save raw (pre-correction) SRT
    raw_merged = transcriber.merge_transcriptions([
        {**t, "segments": t.get("raw_segments", t.get("segments", []))}
        for t in transcriptions
    ])
    raw_srt_path = output_dir / f"{video_path.stem}_raw.srt"
    transcriber.save_transcription_srt(raw_merged, raw_srt_path)

    # Save full transcription with timestamps
    transcript_path = output_dir / f"{video_path.stem}_transcript.txt"
    transcriber.save_transcription(merged, transcript_path, include_timestamps=True)

    # Save plain text version
    plain_text_path = output_dir / f"{video_path.stem}_transcript_plain.txt"
    transcriber.save_transcription(merged, plain_text_path, include_timestamps=False)

    # Save SRT subtitle file
    srt_path = output_dir / f"{video_path.stem}_transcript.srt"
    transcriber.save_transcription_srt(merged, srt_path)

    # Clean up intermediate files if requested
    if not keep_clips:
        click.echo("\nCleaning up intermediate files...")
        import shutil
        shutil.rmtree(clips_dir, ignore_errors=True)
        audio_path.unlink(missing_ok=True)

    # Chapter detection
    chapter_results = None
    if chapters:
        ch_model = chapter_model or correction_model
        if not ch_model:
            click.echo("\nWarning: --chapters requires --chapter-model or --correction-model. Skipping chapter detection.", err=True)
        else:
            click.echo(f"\nStep 5: Detecting chapters...")
            try:
                from .utils import get_video_duration
                video_duration = get_video_duration(video_path)
            except Exception:
                # Fallback: use last segment end time
                last_seg = merged['segments'][-1] if merged['segments'] else {}
                video_duration = last_seg.get('end', 0)

            segmenter = ChapterSegmenter(
                model=ch_model,
                target_duration=chapter_duration,
            )
            chapter_results = segmenter.detect_chapters(
                merged['segments'], video_duration
            )

            if chapter_results:
                chapters_md_path = output_dir / f"{video_path.stem}_chapters.md"
                chapters_json_path = output_dir / f"{video_path.stem}_chapters.json"
                segmenter.save_chapters_md(
                    chapter_results, chapters_md_path,
                    video_name=video_path.stem, total_duration=video_duration,
                )
                segmenter.save_chapters_json(
                    chapter_results, chapters_json_path,
                    video_name=video_path.stem, total_duration=video_duration,
                )
                chapters_csv_path = output_dir / f"{video_path.stem}_chapters_markers.csv"
                chapters_edl_path = output_dir / f"{video_path.stem}_chapters.edl"
                segmenter.save_chapters_csv(
                    chapter_results, chapters_csv_path,
                    video_name=video_path.stem, total_duration=video_duration, fps=fps,
                )
                segmenter.save_chapters_edl(
                    chapter_results, chapters_edl_path,
                    video_name=video_path.stem, total_duration=video_duration, fps=fps,
                )

    # Print summary
    click.echo(f"\n{'='*60}")
    click.echo("Transcription Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"\nResults saved to: {output_dir}")
    click.echo(f"  - Raw SRT (pre-correction): {raw_srt_path.name}")
    click.echo(f"  - Transcript (with timestamps): {transcript_path.name}")
    click.echo(f"  - Transcript (plain text): {plain_text_path.name}")
    click.echo(f"  - SRT subtitle file: {srt_path.name}")
    if chapter_results:
        click.echo(f"  - Chapters (markdown): {video_path.stem}_chapters.md")
        click.echo(f"  - Chapters (JSON): {video_path.stem}_chapters.json")
        click.echo(f"  - Chapters (CSV markers): {video_path.stem}_chapters_markers.csv")
        click.echo(f"  - Chapters (EDL): {video_path.stem}_chapters.edl")
        click.echo(f"  Total chapters: {len(chapter_results)}")
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
@click.option(
    '--initial-prompt',
    type=str,
    default=None,
    help='Initial prompt for Whisper (domain vocabulary, proper nouns, style hints)'
)
@click.option(
    '--prompt-file',
    type=click.Path(exists=True),
    default=None,
    help='Load initial prompt from a text file'
)
@click.option(
    '--beam-size',
    type=int,
    default=5,
    help='Beam search width (default: 5)'
)
@click.option(
    '--compute-type',
    type=click.Choice(['default', 'float16', 'int8_float16', 'int8', 'float32']),
    default='default',
    help='CTranslate2 compute type (default: auto-select)'
)
@click.option(
    '--no-hallucination-filter',
    is_flag=True,
    help='Disable hallucination filtering'
)
@click.option(
    '--vocab-file',
    type=click.Path(exists=True),
    default=None,
    help='Vocabulary file with hotwords and known error examples for LLM correction'
)
@click.option(
    '--max-segment-duration',
    type=int,
    default=10,
    help='Max subtitle segment duration in seconds (default: 10)'
)
@click.option(
    '--correction-model',
    type=str,
    default=None,
    help='LLM model for vocab-based correction (e.g. gemini/gemini-2.0-flash, ollama/qwen3.5:27b, transformers/Qwen/Qwen3.5-27B). Requires --vocab-file'
)
@click.pass_context
def transcribe_audio(ctx, audio_path, output_dir, model, language,
                     initial_prompt, prompt_file, beam_size, compute_type, no_hallucination_filter,
                     vocab_file, max_segment_duration, correction_model):
    """
    Transcribe an audio file directly (without clipping).

    This command transcribes an audio file using Whisper without
    performing VAD-based clipping.
    """
    # Resolve config
    config = ctx.obj.get('config', {})
    vocab_file = vocab_file or config.get('correction', {}).get('vocab_file')
    correction_model = correction_model or config.get('correction', {}).get('model')

    audio_path = Path(audio_path)
    output_dir = Path(output_dir) / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n{'='*60}")
    click.echo(f"LongVideoHelper - Audio Transcription")
    click.echo(f"{'='*60}\n")
    click.echo(f"Audio: {audio_path}")
    click.echo(f"Output Directory: {output_dir}")
    click.echo(f"Whisper Model: {model}")
    click.echo(f"Language: {language or 'auto-detect'}")

    # Load prompt from file if specified
    if prompt_file:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            initial_prompt = f.read().strip()
    if initial_prompt:
        click.echo(f"Initial Prompt: {initial_prompt[:80]}{'...' if len(initial_prompt) > 80 else ''}")
    click.echo()

    click.echo("Transcribing audio...")
    transcriber = Transcriber(
        model_name=model,
        compute_type=compute_type,
        beam_size=beam_size,
        initial_prompt=initial_prompt,
        hallucination_filter=not no_hallucination_filter,
        vocab_file=vocab_file,
        max_segment_duration=max_segment_duration,
        correction_model=correction_model,
    )

    try:
        result = transcriber.transcribe_clip(audio_path, language=language)
    except Exception as e:
        click.echo(f"Error transcribing audio: {str(e)}", err=True)
        return

    # Save raw (pre-correction) SRT
    raw_result = {**result, "segments": result.get("raw_segments", result.get("segments", []))}
    raw_srt_path = output_dir / f"{audio_path.stem}_raw.srt"
    transcriber.save_transcription_srt(raw_result, raw_srt_path)

    # Save results
    transcript_path = output_dir / f"{audio_path.stem}_transcript.txt"
    transcriber.save_transcription(result, transcript_path, include_timestamps=True)

    plain_text_path = output_dir / f"{audio_path.stem}_transcript_plain.txt"
    transcriber.save_transcription(result, plain_text_path, include_timestamps=False)

    srt_path = output_dir / f"{audio_path.stem}_transcript.srt"
    transcriber.save_transcription_srt(result, srt_path)

    # Print summary
    click.echo(f"\n{'='*60}")
    click.echo("Transcription Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"\nResults saved to:")
    click.echo(f"  - Transcript (with timestamps): {transcript_path}")
    click.echo(f"  - Transcript (plain text): {plain_text_path}")
    click.echo(f"  - SRT subtitle file: {srt_path}")
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
    output_dir = Path(output_dir) / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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
@click.argument('transcript_path', type=click.Path(exists=True))
@click.option(
    '--model',
    '-m',
    default=None,
    help='LLM model for chapter detection (e.g. gemini/gemini-2.0-flash, ollama/gpt-oss:20b)'
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default=None,
    help='Output directory (default: same directory as transcript)'
)
@click.option(
    '--duration',
    '-d',
    type=float,
    default=None,
    help='Total video duration in seconds (auto-detected from transcript if not specified)'
)
@click.option(
    '--chapter-duration',
    type=int,
    default=None,
    help='Target chapter duration in seconds (default: 300 = 5 minutes)'
)
@click.option(
    '--min-chapter-duration',
    type=int,
    default=None,
    help='Minimum chapter duration in seconds (default: 120 = 2 minutes)'
)
@click.option(
    '--max-chapter-duration',
    type=int,
    default=None,
    help='Maximum chapter duration in seconds (default: 900 = 15 minutes)'
)
@click.option(
    '--fps',
    type=int,
    default=None,
    help='Frame rate for timecode export (default: 24)'
)
@click.pass_context
def detect_chapters(ctx, transcript_path, model, output_dir, duration,
                    chapter_duration, min_chapter_duration, max_chapter_duration, fps):
    """
    Detect chapter boundaries from an existing transcript file.

    Takes a transcript file (with timestamps) and uses an LLM to identify
    logical chapter boundaries. Useful for re-segmenting without re-transcribing.
    """
    # Resolve config
    config = ctx.obj.get('config', {})
    model = model or config.get('chapters', {}).get('model')
    if not model:
        click.echo("Error: --model is required (or set [chapters] model in config)", err=True)
        return
    fps = fps or config.get('output', {}).get('fps', 24)
    chapter_duration = chapter_duration or config.get('chapters', {}).get('duration', 300)
    min_chapter_duration = min_chapter_duration or 120
    max_chapter_duration = max_chapter_duration or 900

    transcript_path = Path(transcript_path)

    # Determine output directory
    if output_dir is None:
        output_dir = config.get('output', {}).get('dir')
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = transcript_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"\n{'='*60}")
    click.echo(f"LongVideoHelper - Chapter Detection")
    click.echo(f"{'='*60}\n")
    click.echo(f"Transcript: {transcript_path}")
    click.echo(f"Model: {model}")
    click.echo(f"Target chapter duration: {chapter_duration}s ({chapter_duration // 60}m)")

    # Load transcript
    click.echo(f"\nLoading transcript...")
    try:
        transcript_data = parse_transcript_file(transcript_path)
    except Exception as e:
        click.echo(f"Error loading transcript: {e}", err=True)
        return

    segments = transcript_data.get('segments', [])
    if not segments:
        click.echo("Error: No segments found in transcript file.", err=True)
        return

    click.echo(f"Loaded {len(segments)} segments")

    # Determine total duration
    if duration is None:
        duration = segments[-1].get('end', 0)
    click.echo(f"Duration: {duration:.1f}s ({duration / 60:.1f} min)\n")

    # Detect chapters
    segmenter = ChapterSegmenter(
        model=model,
        target_duration=chapter_duration,
        min_duration=min_chapter_duration,
        max_duration=max_chapter_duration,
    )

    chapter_results = segmenter.detect_chapters(segments, duration)

    if not chapter_results:
        click.echo("No chapters detected.", err=True)
        return

    # Save outputs
    stem = transcript_path.stem.replace('_transcript', '').replace('_raw', '')
    chapters_md_path = output_dir / f"{stem}_chapters.md"
    chapters_json_path = output_dir / f"{stem}_chapters.json"

    segmenter.save_chapters_md(
        chapter_results, chapters_md_path,
        video_name=stem, total_duration=duration,
    )
    segmenter.save_chapters_json(
        chapter_results, chapters_json_path,
        video_name=stem, total_duration=duration,
    )

    chapters_csv_path = output_dir / f"{stem}_chapters_markers.csv"
    chapters_edl_path = output_dir / f"{stem}_chapters.edl"
    segmenter.save_chapters_csv(
        chapter_results, chapters_csv_path,
        video_name=stem, total_duration=duration, fps=fps,
    )
    segmenter.save_chapters_edl(
        chapter_results, chapters_edl_path,
        video_name=stem, total_duration=duration, fps=fps,
    )

    # Print summary
    click.echo(f"\n{'='*60}")
    click.echo("Chapter Detection Complete!")
    click.echo(f"{'='*60}")
    click.echo(f"\nResults saved to:")
    click.echo(f"  - Chapters (markdown): {chapters_md_path}")
    click.echo(f"  - Chapters (JSON): {chapters_json_path}")
    click.echo(f"  - Chapters (CSV markers): {chapters_csv_path}")
    click.echo(f"  - Chapters (EDL): {chapters_edl_path}")
    click.echo(f"\nTotal chapters: {len(chapter_results)}\n")


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
@click.option(
    '--initial-prompt',
    type=str,
    default=None,
    help='Initial prompt for Whisper (domain vocabulary, proper nouns, style hints)'
)
@click.option(
    '--prompt-file',
    type=click.Path(exists=True),
    default=None,
    help='Load initial prompt from a text file'
)
@click.option(
    '--beam-size',
    type=int,
    default=5,
    help='Beam search width (default: 5)'
)
@click.option(
    '--compute-type',
    type=click.Choice(['default', 'float16', 'int8_float16', 'int8', 'float32']),
    default='default',
    help='CTranslate2 compute type (default: auto-select)'
)
@click.option(
    '--no-hallucination-filter',
    is_flag=True,
    help='Disable hallucination filtering'
)
@click.option(
    '--vocab-file',
    type=click.Path(exists=True),
    default=None,
    help='Vocabulary file with hotwords and known error examples for LLM correction'
)
@click.option(
    '--max-segment-duration',
    type=int,
    default=10,
    help='Max subtitle segment duration in seconds (default: 10)'
)
@click.option(
    '--correction-model',
    type=str,
    default=None,
    help='LLM model for vocab-based correction (e.g. gemini/gemini-2.0-flash, ollama/qwen3.5:27b, transformers/Qwen/Qwen3.5-27B). Requires --vocab-file'
)
def process(video_path, output_dir, whisper_model, vlm_provider, vlm_model, api_key, language, chapter_duration,
            initial_prompt, prompt_file, beam_size, compute_type, no_hallucination_filter,
            vocab_file, max_segment_duration, correction_model):
    """
    Full pipeline: transcribe video and create chapter summaries.

    This command runs both Phase 1 (transcription) and Phase 2 (chapter creation)
    in a single workflow.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir) / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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
    click.echo(f"VLM: {vlm_model_normalized}")

    # Load prompt from file if specified
    if prompt_file:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            initial_prompt = f.read().strip()
    if initial_prompt:
        click.echo(f"Initial Prompt: {initial_prompt[:80]}{'...' if len(initial_prompt) > 80 else ''}")
    click.echo()

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
    transcriber = Transcriber(
        model_name=whisper_model,
        compute_type=compute_type,
        beam_size=beam_size,
        initial_prompt=initial_prompt,
        hallucination_filter=not no_hallucination_filter,
        vocab_file=vocab_file,
        max_segment_duration=max_segment_duration,
        correction_model=correction_model,
        video_path=str(video_path),
    )

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

    # Save SRT subtitle file
    srt_path = output_dir / f"{video_path.stem}_transcript.srt"
    transcriber.save_transcription_srt(merged, srt_path)

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
    click.echo(f"  - SRT Subtitle: {srt_path}")
    click.echo(f"  - Chapter Summary: {markdown_path}")
    click.echo(f"  - JSON Data: {json_path}")
    click.echo(f"\nTotal chapters: {len(results)}\n")


@cli.command()
@click.option('--output', '-o', type=click.Path(), default='longvideohelper.toml', help='Config file path')
def init_config(output):
    """Generate a template configuration file."""
    output = Path(output)
    if output.exists():
        click.echo(f"Config file already exists: {output}")
        click.echo("Delete it first or use a different path.")
        return

    template = """# LongVideoHelper Configuration
# Place this file in your project directory as longvideohelper.toml

[transcription]
# Whisper model: tiny, base, small, medium, large, turbo
model = "turbo"
# Language code (e.g., "zh", "en") or omit for auto-detect
# language = "zh"
# Beam search width
beam_size = 5

[correction]
# LLM model for vocabulary-based correction
# model = "ollama/gpt-oss:20b"
# Path to vocabulary file
# vocab_file = "vocab_oncehuman.txt"

[chapters]
# Enable chapter detection after transcription
enabled = false
# LLM model for chapter detection
# model = "ollama/gpt-oss:20b"
# Target chapter duration in seconds
duration = 300

[output]
# Output directory
dir = "output"
# Frame rate for timecode export (CSV/EDL)
fps = 24
"""

    with open(output, 'w', encoding='utf-8') as f:
        f.write(template)

    click.echo(f"Config file created: {output}")
    click.echo("Edit the file to customize your settings.")


if __name__ == '__main__':
    cli()
