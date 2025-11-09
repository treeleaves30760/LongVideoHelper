"""Main CLI module for LongVideoHelper."""

import click
from pathlib import Path
from .audio_extractor import AudioExtractor
from .audio_clipper import AudioClipper
from .transcriber import Transcriber


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


if __name__ == '__main__':
    cli()
