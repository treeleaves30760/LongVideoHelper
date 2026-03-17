"""Module for transcribing audio using faster-whisper."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Union, Optional
from tqdm import tqdm
from faster_whisper import WhisperModel
from litellm import completion

logger = logging.getLogger(__name__)


class Transcriber:
    """Handles audio transcription using faster-whisper (CTranslate2)."""

    HALLUCINATION_PHRASES = [
        "thank you for watching",
        "thanks for watching",
        "please subscribe",
        "谢谢观看",
        "感谢观看",
        "谢谢大家",
        "请订阅",
        "字幕由",
        "字幕制作",
    ]

    def __init__(
        self,
        model_name: str = "turbo",
        device: Optional[str] = None,
        compute_type: str = "default",
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        hallucination_filter: bool = True,
        vocab_file: Optional[str] = None,
        max_segment_duration: int = 10,
        correction_model: Optional[str] = None,
        video_path: Optional[str] = None,
    ):
        """
        Initialize the Transcriber.

        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large, turbo)
            device: Device to run the model on ("cuda", "cpu", or None for auto)
            compute_type: CTranslate2 quantization (default, float16, int8_float16, int8, float32)
            beam_size: Beam search width (default: 5)
            initial_prompt: Domain vocabulary / style hints to condition Whisper
            hallucination_filter: Whether to filter likely hallucinated segments
            vocab_file: Path to vocabulary file with hotwords and known error examples
            max_segment_duration: Max subtitle segment duration in seconds (default: 10)
            correction_model: LLM model for vocabulary-based correction.
                For litellm backends: gemini/gemini-2.0-flash, ollama/qwen3.5:27b, etc.
                For local transformers: transformers/Qwen/Qwen3.5-27B
            video_path: Path to video file for OCR-assisted correction
        """
        self.model_name = model_name
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.hallucination_filter = hallucination_filter
        self.max_segment_duration = max_segment_duration
        self.correction_model = correction_model
        self.video_path = Path(video_path) if video_path else None
        self._video_clip = None
        self._ocr = None
        self.hotwords: Optional[str] = None
        self.vocab_terms: List[str] = []
        self.vocab_corrections: List[tuple] = []

        # Determine correction backend
        self._correction_backend = None
        if correction_model:
            if correction_model.startswith("transformers/"):
                self._correction_backend = "transformers"
                self._tf_model_name = correction_model[len("transformers/"):]
                self._tf_model = None
                self._tf_processor = None
                self._tf_load_failed = False
            else:
                self._correction_backend = "litellm"

        if vocab_file:
            self._load_vocab_file(vocab_file)

        logger.info(f"Loading faster-whisper model: {model_name} (compute_type={compute_type})")
        self.model = WhisperModel(
            model_name,
            device=device or "auto",
            compute_type=compute_type,
        )
        logger.info("Model loaded successfully")

    def _load_vocab_file(self, vocab_file: str) -> None:
        """Load hotwords and known error examples from a vocabulary file.

        Format:
            # comments
            詞彙A          → hotword + LLM reference term
            詞彙B          → hotword + LLM reference term
            錯誤->正確     → known error example for LLM context
        """
        terms = []
        corrections = []

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '->' in line:
                    parts = line.split('->', 1)
                    corrections.append((parts[0].strip(), parts[1].strip()))
                else:
                    terms.append(line)

        if terms:
            self.vocab_terms = terms
            # Whisper hotwords: limit to 20 to avoid crashes with large vocab
            MAX_HOTWORDS = 20
            self.hotwords = " ".join(terms[:MAX_HOTWORDS])
            if len(terms) > MAX_HOTWORDS:
                logger.info(f"Loaded {len(terms)} vocabulary terms from {vocab_file} (hotwords limited to {MAX_HOTWORDS})")
            else:
                logger.info(f"Loaded {len(terms)} vocabulary terms from {vocab_file}")
        if corrections:
            self.vocab_corrections = corrections
            logger.info(f"Loaded {len(corrections)} known error examples from {vocab_file}")

    def _get_ocr(self):
        """Lazy-load PaddleOCR on first use."""
        if self._ocr is None:
            # Pre-import torch to avoid DLL loading issues on Windows
            try:
                import torch
            except ImportError:
                pass
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang='chinese_cht', show_log=False)
            logger.info("PaddleOCR loaded (chinese_cht)")
        return self._ocr

    def _get_video_clip(self):
        """Lazy-load and cache the video clip."""
        if self._video_clip is None:
            from moviepy import VideoFileClip
            self._video_clip = VideoFileClip(str(self.video_path))
        return self._video_clip

    def _close_video_clip(self):
        """Close the cached video clip to free resources."""
        if self._video_clip is not None:
            self._video_clip.close()
            self._video_clip = None

    def _extract_batch_frames(self, batch: List[Dict], clip_offset: float) -> tuple:
        """Extract frames and run OCR for a correction batch.

        Returns: (frame_paths, ocr_texts)
          - frame_paths: list of JPEG file paths (for VLM image input)
          - ocr_texts: list of unique OCR-extracted text strings
        """
        import tempfile
        try:
            clip = self._get_video_clip()
            ocr = self._get_ocr()

            # Calculate batch time range in video time
            batch_start = batch[0]["start"] + clip_offset
            batch_end = batch[-1]["end"] + clip_offset
            batch_start = max(0.0, batch_start)
            batch_end = min(clip.duration, batch_end)

            # Extract one frame every 10 seconds
            frame_times = []
            t = batch_start
            while t <= batch_end:
                frame_times.append(t)
                t += 10.0
            if not frame_times:
                frame_times = [batch_start]

            tmpdir = tempfile.mkdtemp(prefix="lvh_frames_")
            frame_paths = []
            all_ocr_texts = set()

            import cv2
            import numpy as np
            for i, t in enumerate(frame_times):
                try:
                    frame = clip.get_frame(t)
                    frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                    frame_path = Path(tmpdir) / f"frame_{i:03d}.jpg"
                    cv2.imwrite(str(frame_path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    frame_paths.append(frame_path)

                    # Run OCR on the frame (confidence threshold 0.6)
                    result = ocr.ocr(str(frame_path), cls=True)
                    if result and result[0]:
                        for line in result[0]:
                            text = line[1][0].strip()
                            conf = line[1][1]
                            if text and conf >= 0.6:
                                all_ocr_texts.add(text)
                except Exception as e:
                    logger.debug(f"Frame extraction/OCR failed at t={t:.1f}s: {e}")
                    continue

            # Filter OCR texts: keep only meaningful text (≥2 Chinese chars, no pure numbers/codes)
            filtered_texts = set()
            for text in all_ocr_texts:
                # Count Chinese characters
                chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
                if chinese_chars >= 2:
                    filtered_texts.add(text)

            ocr_texts = sorted(filtered_texts)
            logger.info(f"Extracted {len(frame_paths)} frames, OCR found {len(ocr_texts)} unique texts (filtered from {len(all_ocr_texts)})")
            return frame_paths, ocr_texts

        except Exception as e:
            logger.warning(f"Batch frame extraction failed, falling back to text-only: {e}")
            return [], []

    def _build_correction_prompt(self, texts: List[str], has_frames: bool = False, ocr_texts: Optional[List[str]] = None) -> str:
        """Build the correction prompt for a batch of texts.

        Uses a simple numbered format that small models can follow reliably.
        """
        vocab_str = "、".join(self.vocab_terms)

        corrections_ctx = ""
        if self.vocab_corrections:
            examples = " / ".join(f"{w}→{c}" for w, c in self.vocab_corrections[:5])
            corrections_ctx = f"\n常見錯誤：{examples}"

        ocr_ctx = ""
        if ocr_texts:
            ocr_joined = "、".join(ocr_texts[:50])
            ocr_ctx = f"\n畫面 OCR 文字：{ocr_joined}"

        visual_hint = ""
        if has_frames:
            visual_hint = "\n參考畫面中可見的遊戲介面文字來判斷正確用詞。"

        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

        return (
            f"修正以下字幕中的專有名詞錯誤。\n"
            f"參考詞彙：{vocab_str}{corrections_ctx}{ocr_ctx}\n"
            f"規則：只改錯字，不改語意，無錯則原文輸出。{visual_hint}\n\n"
            f"{numbered}\n\n"
            f"請輸出修正後的 {len(texts)} 行，格式完全相同（編號. 內容）："
        )

    def _parse_correction_response(self, result_text: str, original_texts: List[str]) -> List[Optional[str]]:
        """Parse LLM correction response, returning corrected texts or None for failures."""
        n = len(original_texts)
        corrected = [None] * n

        # Strip thinking tags (Qwen3.5 thinking mode)
        result_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL).strip()
        # Strip markdown headers that models sometimes output
        result_text = re.sub(r'\*\*[^*]+\*\*:?\s*', '', result_text).strip()

        # Try numbered line format: "1. text", "2. text", ...
        lines = result_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            m = re.match(r'^(\d+)\.\s*(.+)$', line)
            if m:
                idx = int(m.group(1)) - 1
                text = m.group(2).strip()
                if 0 <= idx < n:
                    corrected[idx] = text

        # Count how many we parsed
        parsed = sum(1 for c in corrected if c is not None)
        if parsed < n * 0.5:
            # Less than half parsed — try JSON fallback
            json_match = re.search(r'\[.*?\]', result_text, re.DOTALL)
            if json_match:
                try:
                    arr = json.loads(json_match.group(0))
                    if isinstance(arr, list) and len(arr) == n:
                        for i, item in enumerate(arr):
                            if isinstance(item, str) and item.strip():
                                corrected[i] = item.strip()
                except (json.JSONDecodeError, ValueError):
                    pass

        return corrected

    def _is_valid_correction(self, original: str, corrected: str) -> bool:
        """Check if a correction is reasonable (not garbage)."""
        if not corrected:
            return False
        orig = original.replace(" ", "")
        corr = corrected.replace(" ", "")
        orig_len = len(orig)
        corr_len = len(corr)
        if not corr_len:
            return False
        # Reject if length changed too drastically (>2x or <0.3x)
        if orig_len > 0 and (corr_len > orig_len * 2 or corr_len < orig_len * 0.3):
            return False
        # Reject if corrected text has very low character diversity (e.g. "消消消消")
        unique_chars = len(set(corr))
        if corr_len > 5 and unique_chars <= 2:
            return False
        # Reject placeholder patterns
        if re.match(r'^string\d*$', corrected.strip(), re.IGNORECASE):
            return False
        # Reject if correction shares too few characters with original (must overlap >40%)
        if orig_len > 3:
            shared = sum(1 for c in corr if c in orig)
            if shared / max(orig_len, corr_len) < 0.4:
                return False
        return True

    def _get_transformers_model(self):
        """Lazy-load the transformers model on first use."""
        if self._tf_load_failed:
            raise RuntimeError("Transformers model failed to load previously")
        if self._tf_model is None:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText

            try:
                logger.info(f"Loading transformers model: {self._tf_model_name} (bfloat16, GPU)...")
                self._tf_processor = AutoProcessor.from_pretrained(self._tf_model_name)
                self._tf_model = AutoModelForImageTextToText.from_pretrained(
                    self._tf_model_name,
                    dtype=torch.bfloat16,
                    device_map="cuda",
                )
                logger.info(f"Transformers model loaded (~{torch.cuda.memory_allocated()/1e9:.1f}GB VRAM)")
            except Exception as e:
                self._tf_load_failed = True
                raise
        return self._tf_model, self._tf_processor

    def _call_litellm(self, prompt: str, image_paths: Optional[List[Path]] = None) -> str:
        """Call LLM via litellm and return response text."""
        if image_paths:
            import base64
            content = [{"type": "text", "text": prompt}]
            for img_path in image_paths:
                if not img_path.exists():
                    continue
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                img_type = img_path.suffix.lower().replace('.', '')
                if img_type == 'jpg':
                    img_type = 'jpeg'
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{img_type};base64,{img_data}"}
                })
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": prompt}]

        response = completion(
            model=self.correction_model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content

    def _call_transformers(self, prompt: str, image_paths: Optional[List[Path]] = None) -> str:
        """Call local transformers model and return response text."""
        model, processor = self._get_transformers_model()

        content = []
        if image_paths:
            try:
                from PIL import Image as PILImage
                for img_path in image_paths:
                    if img_path.exists():
                        content.append({"type": "image", "image": PILImage.open(str(img_path))})
            except Exception as e:
                logger.debug(f"Failed to load images for transformers, using text-only: {e}")
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        template_kwargs = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Disable thinking/reasoning mode for Qwen models that support it
        if "qwen" in self._tf_model_name.lower():
            try:
                # Test if the template supports enable_thinking
                test_result = processor.apply_chat_template(
                    [{"role": "user", "content": "test"}],
                    add_generation_prompt=True, tokenize=False,
                    enable_thinking=False,
                )
                template_kwargs["enable_thinking"] = False
            except (TypeError, ValueError):
                pass

        inputs = processor.apply_chat_template(
            messages,
            **template_kwargs,
        ).to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        return processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def _unload_whisper(self):
        """Temporarily unload Whisper model to free GPU memory for correction."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.debug("Whisper model unloaded to free VRAM")

    def _reload_whisper(self):
        """Reload Whisper model after correction."""
        if self.model is None:
            self.model = WhisperModel(
                self.model_name,
                device="auto",
                compute_type="default",
            )
            logger.debug("Whisper model reloaded")

    def _correct_segments_with_llm(self, segments: List[Dict], clip_offset: float = 0.0) -> List[Dict]:
        """Use LLM to correct misrecognized terms in segments based on vocabulary reference."""
        if not self.vocab_terms or not self._correction_backend or not segments:
            return segments

        BATCH_SIZE = 20
        total_batches = (len(segments) + BATCH_SIZE - 1) // BATCH_SIZE
        total_changed = 0
        total_failed = 0

        logger.info(f"Correcting {len(segments)} segments with {self.correction_model} ({total_batches} batches, size={BATCH_SIZE})")

        for batch_start in range(0, len(segments), BATCH_SIZE):
            batch = segments[batch_start:batch_start + BATCH_SIZE]
            texts = [seg["text"].strip() for seg in batch]
            batch_num = batch_start // BATCH_SIZE + 1

            # Extract frames and OCR if video is available
            frame_paths = []
            ocr_texts = []
            if self.video_path:
                frame_paths, ocr_texts = self._extract_batch_frames(batch, clip_offset)

            # Only pass images to vision-capable models (litellm API models)
            # Text-only transformers models get OCR text only
            send_images = frame_paths and self._correction_backend == "litellm"

            prompt = self._build_correction_prompt(
                texts,
                has_frames=send_images,
                ocr_texts=ocr_texts or None,
            )

            try:
                if self._correction_backend == "transformers":
                    result_text = self._call_transformers(prompt)
                else:
                    result_text = self._call_litellm(prompt, image_paths=frame_paths if send_images else None)

                corrected = self._parse_correction_response(result_text, texts)
                changed = 0
                for i, seg in enumerate(batch):
                    if corrected[i] is not None and corrected[i] != texts[i]:
                        if self._is_valid_correction(texts[i], corrected[i]):
                            logger.debug(f"  [{i}] '{texts[i][:30]}' -> '{corrected[i][:30]}'")
                            seg["text"] = corrected[i]
                            changed += 1
                        else:
                            logger.debug(f"  [{i}] Rejected bad correction: '{corrected[i][:30]}'")

                if changed > 0:
                    logger.info(f"Batch {batch_num}/{total_batches}: {changed} corrected")
                total_changed += changed

            except Exception as e:
                total_failed += 1
                logger.warning(f"Batch {batch_num}/{total_batches} failed: {e}")
            finally:
                # Clean up temporary frame files
                for fp in frame_paths:
                    try:
                        fp.unlink(missing_ok=True)
                    except Exception:
                        pass
                # Clean up temp directory if it exists and is empty
                if frame_paths:
                    try:
                        frame_paths[0].parent.rmdir()
                    except Exception:
                        pass

        self._close_video_clip()

        # Rule-based post-correction: apply known error→correct mappings deterministically
        if self.vocab_corrections:
            rule_changed = 0
            for seg in segments:
                text = seg["text"]
                for wrong, correct in self.vocab_corrections:
                    if wrong in text:
                        text = text.replace(wrong, correct)
                if text != seg["text"]:
                    logger.debug(f"  Rule fix: '{seg['text'][:40]}' -> '{text[:40]}'")
                    seg["text"] = text
                    rule_changed += 1
            if rule_changed:
                total_changed += rule_changed
                logger.info(f"Rule-based post-correction: {rule_changed} additional fixes")

        logger.info(f"Correction done: {total_changed} segments changed, {total_failed} batches failed")
        return segments

    def _split_long_segments(self, segments: List[Dict], max_duration: float) -> List[Dict]:
        """Split segments longer than max_duration at natural boundaries."""
        PUNCTUATION = set("。，！？、,.!?;；")
        result = []

        for seg in segments:
            duration = seg["end"] - seg["start"]
            words = seg.get("words")

            if duration <= max_duration or not words or len(words) < 2:
                result.append(seg)
                continue

            # Find split points
            split_indices = []
            last_split_time = seg["start"]

            for i, w in enumerate(words):
                elapsed = w["end"] - last_split_time
                if elapsed >= max_duration and i < len(words) - 1:
                    # Look backward for a punctuation boundary
                    best = i
                    for j in range(i, max(i - 5, 0) - 1, -1):
                        if words[j]["word"].rstrip() and words[j]["word"].rstrip()[-1] in PUNCTUATION:
                            best = j
                            break
                    split_indices.append(best)
                    last_split_time = words[best + 1]["start"] if best + 1 < len(words) else w["end"]

            if not split_indices:
                result.append(seg)
                continue

            # Build sub-segments from split points
            boundaries = [0] + [si + 1 for si in split_indices] + [len(words)]
            for k in range(len(boundaries) - 1):
                chunk = words[boundaries[k]:boundaries[k + 1]]
                if not chunk:
                    continue
                sub = {
                    "start": chunk[0]["start"],
                    "end": chunk[-1]["end"],
                    "text": "".join(w["word"] for w in chunk),
                    "words": chunk,
                }
                # Carry over metadata from original segment
                if "no_speech_prob" in seg:
                    sub["no_speech_prob"] = seg["no_speech_prob"]
                if "avg_logprob" in seg:
                    sub["avg_logprob"] = seg["avg_logprob"]
                result.append(sub)

        return result

    def _is_hallucination(self, text: str, no_speech_prob: float, avg_logprob: float) -> bool:
        """Check if a segment is likely a hallucination."""
        if not self.hallucination_filter:
            return False

        if no_speech_prob > 0.9:
            return True

        if avg_logprob < -1.5:
            return True

        text_lower = text.strip().lower()

        if not text_lower:
            return True

        for phrase in self.HALLUCINATION_PHRASES:
            if phrase in text_lower:
                return True

        # Detect repetitive patterns
        if len(text_lower) > 10:
            # Single-character: require 5+ consecutive (好好好 is common in Chinese)
            for i in range(len(text_lower) - 4):
                c = text_lower[i]
                if c.strip() and text_lower[i:i + 5] == c * 5:
                    return True
            # Multi-character (2+): require 3+ consecutive
            for length in range(2, min(20, len(text_lower) // 3 + 1)):
                for i in range(len(text_lower) - length * 3 + 1):
                    chunk = text_lower[i:i + length]
                    if chunk.strip() and text_lower[i:i + length * 3] == chunk * 3:
                        return True

        return False

    def _run_whisper(self, audio_path: Path, language, prompt) -> tuple:
        """Run Whisper transcription with automatic fallback on crash.

        If transcription crashes or covers less than half the audio, retries
        without hotwords and/or initial_prompt.
        """
        from pydub import AudioSegment as PydubSegment
        clip_duration = len(PydubSegment.from_wav(str(audio_path))) / 1000.0

        attempts = [
            dict(initial_prompt=prompt, hotwords=self.hotwords),
            dict(initial_prompt=prompt, hotwords=None),       # drop hotwords
            dict(initial_prompt=None, hotwords=None),          # drop everything
        ]

        best_segments = []
        best_info = None
        best_coverage = 0.0

        for i, overrides in enumerate(attempts):
            transcribe_kwargs = dict(
                language=language,
                beam_size=self.beam_size,
                temperature=0.0,
                initial_prompt=overrides["initial_prompt"],
                condition_on_previous_text=True,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            if overrides.get("hotwords"):
                transcribe_kwargs["hotwords"] = overrides["hotwords"]

            segments_gen, info = self.model.transcribe(str(audio_path), **transcribe_kwargs)

            segments = []
            prev_text = ""
            crashed = False
            try:
                for seg in segments_gen:
                    if seg.end - seg.start < 0.1:
                        continue
                    if self._is_hallucination(seg.text, seg.no_speech_prob, seg.avg_logprob):
                        continue
                    stripped = seg.text.strip()
                    if stripped and stripped == prev_text:
                        continue
                    prev_text = stripped

                    segment_dict = {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "no_speech_prob": seg.no_speech_prob,
                        "avg_logprob": seg.avg_logprob,
                    }
                    if seg.words:
                        segment_dict["words"] = [
                            {"word": w.word, "start": w.start, "end": w.end}
                            for w in seg.words
                        ]
                    segments.append(segment_dict)
            except (ValueError, RuntimeError) as e:
                crashed = True
                logger.debug(f"Whisper crashed (attempt {i+1}): {e}")

            # Calculate quality: sum of actual segment durations / clip duration
            # This avoids being fooled by sparse hallucinated segments at late timestamps
            seg_duration = sum(s["end"] - s["start"] for s in segments) if segments else 0.0
            coverage = seg_duration / clip_duration if clip_duration > 0 else 0.0

            # Keep the best result so far
            if coverage > best_coverage:
                best_segments = segments
                best_info = info
                best_coverage = coverage

            # Good enough: covered >50% of the clip without crash
            if not crashed and coverage > 0.5:
                if i > 0:
                    logger.info(f"Whisper retry succeeded (attempt {i+1}): {len(segments)} segments, {coverage:.0%} coverage")
                return segments, info

            # Crashed but good coverage (>50%): keep it
            if crashed and coverage > 0.5:
                logger.warning(f"Whisper interrupted (attempt {i+1}): {len(segments)} segments, {coverage:.0%} coverage")
                return segments, info

            # Poor coverage: retry with fewer features
            if i < len(attempts) - 1:
                label = "crash" if crashed else f"low coverage ({coverage:.0%})"
                logger.info(f"Whisper attempt {i+1} {label}, {len(segments)} segs — retrying...")

        # Return the best result we found across all attempts
        if best_segments:
            logger.info(f"Whisper best result: {len(best_segments)} segments, {best_coverage:.0%} coverage")
        return best_segments, best_info

    def transcribe_clip(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        clip_offset: float = 0.0,
    ) -> Dict:
        """
        Transcribe a single audio clip.

        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'en', 'zh') or None for auto-detect
            initial_prompt: Override initial prompt for this clip

        Returns:
            Dictionary containing transcription results with keys:
            - text: The transcribed text
            - segments: List of segment dictionaries with timing information
            - language: Detected language
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        prompt = initial_prompt if initial_prompt is not None else self.initial_prompt

        segments, info = self._run_whisper(audio_path, language, prompt)

        logger.debug(f"Transcription produced {len(segments)} segments after filtering (language={info.language})")

        # Split long segments for subtitle readability
        pre_split = len(segments)
        segments = self._split_long_segments(segments, self.max_segment_duration)
        if len(segments) != pre_split:
            logger.debug(f"Segment splitting: {pre_split} -> {len(segments)} segments (max_duration={self.max_segment_duration}s)")

        # Save raw segments before LLM correction
        import copy
        raw_segments = copy.deepcopy(segments)

        # Correct misrecognized terms using LLM
        if self.vocab_terms and self.correction_model:
            self._unload_whisper()
            segments = self._correct_segments_with_llm(segments, clip_offset=clip_offset)
            self._reload_whisper()

        # Rebuild full text from (potentially corrected) segments
        full_text_parts = [seg["text"].strip() for seg in segments]

        return {
            "text": " ".join(full_text_parts),
            "segments": segments,
            "raw_segments": raw_segments,
            "language": info.language,
        }

    def transcribe_clips(
        self,
        clips: List[tuple],
        language: Optional[str] = None,
    ) -> List[Dict]:
        """
        Transcribe multiple audio clips with cross-clip context propagation.

        Args:
            clips: List of tuples (clip_path, start_time, end_time)
            language: Language code or None for auto-detect

        Returns:
            List of transcription results with adjusted timestamps
        """
        results = []
        previous_text = ""

        logger.info(f"Transcribing {len(clips)} audio clips...")

        for clip_path, start_time, end_time in tqdm(clips, desc="Transcribing"):
            try:
                # Combine initial_prompt (domain terms) with trailing context from previous clip
                prompt_parts = []
                if self.initial_prompt:
                    prompt_parts.append(self.initial_prompt)
                if previous_text:
                    prompt_parts.append(previous_text[-200:])
                prompt = " ".join(prompt_parts) if prompt_parts else None

                transcription = self.transcribe_clip(clip_path, language, initial_prompt=prompt, clip_offset=start_time)

                # Adjust segment timestamps to reflect position in original audio
                adjusted_segments = []
                for segment in transcription.get("segments", []):
                    adjusted_segment = segment.copy()
                    adjusted_segment["start"] += start_time
                    adjusted_segment["end"] += start_time
                    if "words" in adjusted_segment:
                        adjusted_segment["words"] = [
                            {**w, "start": w["start"] + start_time, "end": w["end"] + start_time}
                            for w in adjusted_segment["words"]
                        ]
                    adjusted_segments.append(adjusted_segment)

                # Also adjust raw_segments timestamps
                adjusted_raw = []
                for segment in transcription.get("raw_segments", []):
                    adj = segment.copy()
                    adj["start"] += start_time
                    adj["end"] += start_time
                    adjusted_raw.append(adj)

                result = {
                    "clip_path": clip_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": transcription["text"],
                    "segments": adjusted_segments,
                    "raw_segments": adjusted_raw,
                    "language": transcription.get("language", "unknown"),
                }
                results.append(result)

                # Update context for next clip
                if transcription["text"]:
                    previous_text = transcription["text"]

            except Exception as e:
                logger.error(f"Error transcribing {clip_path}: {e}", exc_info=True)
                results.append({
                    "clip_path": clip_path,
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": "",
                    "segments": [],
                    "language": "unknown",
                    "error": str(e),
                })

        return results

    def merge_transcriptions(self, transcriptions: List[Dict]) -> Dict:
        """
        Merge multiple transcriptions into a single result.

        Args:
            transcriptions: List of transcription results

        Returns:
            Merged transcription with:
            - text: Full transcription text
            - segments: All segments combined
            - language: Most common language detected
        """
        full_text = ' '.join([t['text'].strip() for t in transcriptions if t['text']])

        all_segments = []
        for t in transcriptions:
            all_segments.extend(t.get('segments', []))

        # Get most common language
        languages = [t['language'] for t in transcriptions if 'error' not in t]
        language = max(set(languages), key=languages.count) if languages else 'unknown'

        return {
            'text': full_text,
            'segments': all_segments,
            'language': language
        }

    def save_transcription(
        self,
        transcription: Dict,
        output_path: Union[str, Path],
        include_timestamps: bool = True
    ):
        """
        Save transcription to a file.

        Args:
            transcription: Transcription result dictionary
            output_path: Path to save the transcription
            include_timestamps: Whether to include timestamps in the output
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            if include_timestamps and 'segments' in transcription:
                for segment in transcription['segments']:
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '').strip()
                    f.write(f"[{start:.2f} - {end:.2f}] {text}\n")
            else:
                f.write(transcription['text'] + '\n')

        logger.info(f"Transcription saved to: {output_path}")

    def save_transcription_srt(
        self,
        transcription: Dict,
        output_path: Union[str, Path]
    ):
        """
        Save transcription in SRT (SubRip) format for video editing.

        Args:
            transcription: Transcription result dictionary with segments
            output_path: Path to save the SRT file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            if 'segments' in transcription:
                for i, segment in enumerate(transcription['segments'], start=1):
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '').strip()

                    start_time = self._seconds_to_srt_time(start)
                    end_time = self._seconds_to_srt_time(end)

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")

        logger.info(f"SRT file saved to: {output_path}")

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
