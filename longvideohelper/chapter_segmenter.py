"""Module for detecting chapter boundaries and generating summaries from transcripts.

This module provides lightweight LLM-based chapter segmentation that integrates
directly with the transcription pipeline via litellm (same backend as correction).
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

from litellm import completion

logger = logging.getLogger(__name__)


class ChapterSegmenter:
    """Detects chapter boundaries in transcripts using LLM."""

    MAX_TRANSCRIPT_CHARS = 60000

    def __init__(
        self,
        model: str,
        target_duration: int = 300,
        min_duration: int = 120,
        max_duration: int = 900,
    ):
        """
        Initialize the ChapterSegmenter.

        Args:
            model: LLM model for litellm (e.g. gemini/gemini-2.0-flash, ollama/gpt-oss:20b)
            target_duration: Target chapter duration in seconds (default: 300 = 5 min)
            min_duration: Minimum chapter duration in seconds (default: 120 = 2 min)
            max_duration: Maximum chapter duration in seconds (default: 900 = 15 min)
        """
        self.model = model
        self.target_duration = target_duration
        self.min_duration = min_duration
        self.max_duration = max_duration

    @staticmethod
    def _format_ts(seconds: float) -> str:
        """Format seconds as H:MM:SS or M:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    @staticmethod
    def _format_ts_padded(seconds: float) -> str:
        """Format seconds as HH:MM:SS (zero-padded, for YouTube chapters)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}:{m:02d}:{s:02d}"

    @staticmethod
    def _parse_ts(ts: str) -> float:
        """Parse H:MM:SS, MM:SS, or raw seconds to float seconds."""
        ts = ts.strip()
        parts = ts.split(':')
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(ts)
        except (ValueError, IndexError):
            return 0.0

    def _condense_transcript(self, segments: List[Dict]) -> str:
        """Condense transcript by merging segments into time windows.

        Automatically increases window size if transcript is too long.
        """
        if not segments:
            return ""

        for window_seconds in [15, 30, 60, 120]:
            lines = []
            window_start = segments[0].get('start', 0)
            window_texts = []

            for seg in segments:
                seg_start = seg.get('start', 0)
                if seg_start - window_start >= window_seconds and window_texts:
                    ts = self._format_ts(window_start)
                    lines.append(f"[{ts}] {''.join(window_texts)}")
                    window_start = seg_start
                    window_texts = []
                window_texts.append(seg.get('text', '').strip())

            if window_texts:
                ts = self._format_ts(window_start)
                lines.append(f"[{ts}] {''.join(window_texts)}")

            condensed = '\n'.join(lines)
            if len(condensed) <= self.MAX_TRANSCRIPT_CHARS:
                logger.info(
                    f"Condensed: {len(segments)} segments -> {len(lines)} lines "
                    f"({len(condensed)} chars, window={window_seconds}s)"
                )
                return condensed

        logger.warning(f"Transcript still {len(condensed)} chars after 120s windows, truncating")
        return condensed[:self.MAX_TRANSCRIPT_CHARS]

    def _build_prompt(self, condensed: str, total_duration: float) -> str:
        """Build the chapter detection prompt."""
        duration_str = self._format_ts(total_duration)
        num_suggested = max(2, round(total_duration / self.target_duration))
        min_min = self.min_duration // 60
        max_min = self.max_duration // 60

        return f"""你是影片章節分析助手。請分析以下逐字稿，將影片分成合理的章節段落。

**影片資訊：**
- 總長度：{duration_str}
- 建議章節數：約 {num_suggested} 個（可依內容調整）
- 每個章節建議 {min_min}-{max_min} 分鐘

**逐字稿（格式：[時間] 內容）：**
{condensed}

**任務：**
根據內容主題的自然轉換來劃分章節。常見的章節切換點包括：話題改變、開始新活動、場景轉移、重要事件發生等。

**要求：**
1. start 使用逐字稿中最接近的時間戳記
2. 標題簡潔（5-15 字），能讓人快速了解該段主要內容
3. 摘要用一到兩句話描述該章節的重點
4. 確保章節連續覆蓋整個影片，不要有遺漏的時間段
5. 為每個章節標記精彩程度 highlight（1-5，5最精彩如Boss戰、搞笑時刻）
6. 為每個章節標記內容標籤 tags，從以下選擇：戰鬥、Boss戰、探索、建造、搞笑、日常、教學、劇情

**語言：請全程使用繁體中文（zh-TW）輸出。**

**請只輸出以下 JSON 格式，不要輸出任何其他文字：**
{{"chapters": [{{"start": "0:00", "title": "章節標題", "summary": "該章節摘要", "highlight": 3, "tags": ["戰鬥"]}}]}}"""

    def _call_llm(self, prompt: str) -> str:
        """Call LLM via litellm."""
        logger.info(f"Calling {self.model} for chapter detection ({len(prompt)} chars)")

        # Long transcripts need generous timeout (especially for local models)
        # ~20 tokens/sec generation + prompt prefill → allow ~1s per 20 chars + buffer
        timeout = max(1200, len(prompt) // 5)
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            timeout=timeout,
        )
        result = response.choices[0].message.content
        if result is None:
            raise RuntimeError("LLM returned empty response")
        logger.info(f"LLM response: {len(result)} chars")
        return result

    def _parse_response(
        self, response: str, segments: List[Dict], total_duration: float
    ) -> List[Dict]:
        """Parse LLM response into structured chapter list."""
        # Strip thinking tags (Qwen etc.)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Extract JSON from response
        json_str = None

        # Try ```json ... ``` block first
        m = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            # Try raw JSON object
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if m:
                json_str = m.group(0)

        if not json_str:
            raise ValueError(f"No JSON found in LLM response:\n{response[:500]}")

        data = json.loads(json_str)
        raw_chapters = data.get('chapters', [])
        if not raw_chapters:
            raise ValueError("Empty chapters list in response")

        # Parse into structured format
        chapters = []
        for i, ch in enumerate(raw_chapters):
            start_str = str(ch.get('start', '0:00'))
            start_time = self._parse_ts(start_str)
            title = str(ch.get('title', f'段落 {i + 1}')).strip()
            summary = str(ch.get('summary', '')).strip()
            highlight = ch.get('highlight', 3)
            highlight = max(1, min(5, int(highlight) if isinstance(highlight, (int, float)) else 3))
            tags = ch.get('tags', [])
            if not isinstance(tags, list):
                tags = []
            chapters.append({
                'index': i + 1,
                'start_time': start_time,
                'title': title,
                'summary': summary,
                'highlight': highlight,
                'tags': tags,
            })

        # Sort by start time
        chapters.sort(key=lambda c: c['start_time'])

        # Calculate end times (each chapter ends where the next begins)
        for i in range(len(chapters) - 1):
            chapters[i]['end_time'] = chapters[i + 1]['start_time']
        if chapters:
            chapters[-1]['end_time'] = total_duration

        # Validate and fix
        chapters = self._validate(chapters, total_duration)

        # Attach transcript segments to each chapter
        for ch in chapters:
            ch['transcript_segments'] = [
                seg for seg in segments
                if seg.get('end', 0) > ch['start_time'] and seg.get('start', 0) < ch['end_time']
            ]

        return chapters

    def _validate(self, chapters: List[Dict], total_duration: float) -> List[Dict]:
        """Validate and fix chapter boundaries."""
        if not chapters:
            return []

        # Remove entries with duplicate start times
        seen_starts = set()
        unique = []
        for ch in chapters:
            key = round(ch['start_time'], 1)
            if key not in seen_starts:
                seen_starts.add(key)
                unique.append(ch)
        chapters = unique

        # Recalculate end times after dedup
        for i in range(len(chapters) - 1):
            chapters[i]['end_time'] = chapters[i + 1]['start_time']
        if chapters:
            chapters[-1]['end_time'] = total_duration

        # Merge chapters shorter than min_duration into the previous one
        merged = []
        for ch in chapters:
            duration = ch['end_time'] - ch['start_time']
            if duration < self.min_duration and merged:
                merged[-1]['end_time'] = ch['end_time']
                if ch.get('summary'):
                    prev_summary = merged[-1].get('summary', '')
                    merged[-1]['summary'] = f"{prev_summary}；{ch['summary']}" if prev_summary else ch['summary']
                merged[-1]['highlight'] = max(merged[-1].get('highlight', 3), ch.get('highlight', 3))
                for tag in ch.get('tags', []):
                    if tag not in merged[-1].get('tags', []):
                        merged[-1].setdefault('tags', []).append(tag)
            else:
                merged.append(ch)

        # Re-index
        for i, ch in enumerate(merged):
            ch['index'] = i + 1

        return merged

    def _build_chunk_prompt(
        self, condensed: str, chunk_start: float, chunk_end: float, total_duration: float
    ) -> str:
        """Build prompt for a chunk of a longer transcript."""
        chunk_duration = chunk_end - chunk_start
        num_suggested = max(1, round(chunk_duration / self.target_duration))
        min_min = self.min_duration // 60
        max_min = self.max_duration // 60

        return f"""你是影片章節分析助手。以下是一段較長影片中的一個片段，請為這個片段劃分章節。

**片段資訊：**
- 片段時間範圍：{self._format_ts(chunk_start)} - {self._format_ts(chunk_end)}
- 片段長度：{self._format_ts(chunk_duration)}
- 影片總長度：{self._format_ts(total_duration)}
- 建議此片段的章節數：約 {num_suggested} 個（可依內容調整）
- 每個章節建議 {min_min}-{max_min} 分鐘

**逐字稿（格式：[時間] 內容）：**
{condensed}

**任務：**
根據內容主題的自然轉換來劃分章節。常見的章節切換點包括：話題改變、開始新活動、場景轉移、重要事件發生等。

**要求：**
1. start 使用逐字稿中最接近的時間戳記（注意：時間戳是影片的絕對時間）
2. 標題簡潔（5-15 字），能讓人快速了解該段主要內容
3. 摘要用一到兩句話描述該章節的重點
4. 為每個章節標記精彩程度 highlight（1-5，5最精彩如Boss戰、搞笑時刻）
5. 為每個章節標記內容標籤 tags，從以下選擇：戰鬥、Boss戰、探索、建造、搞笑、日常、教學、劇情

**語言：請全程使用繁體中文（zh-TW）輸出。**

**請只輸出以下 JSON 格式，不要輸出任何其他文字：**
{{"chapters": [{{"start": "0:00", "title": "章節標題", "summary": "該章節摘要", "highlight": 3, "tags": ["戰鬥"]}}]}}"""

    def _detect_chunked(
        self, segments: List[Dict], total_duration: float
    ) -> List[Dict]:
        """Detect chapters by processing transcript in sequential chunks.

        For long transcripts (>30 min), processing in chunks yields much better
        results because the LLM can focus on each portion.
        """
        CHUNK_DURATION = 1800  # 30 minutes per chunk
        OVERLAP = 120  # 2 min overlap for context continuity

        all_chapters = []
        first_start = segments[0].get('start', 0)
        chunk_start = first_start

        chunk_num = 0
        effective_step = CHUNK_DURATION - OVERLAP
        total_chunks = max(1, int((total_duration - first_start + effective_step - 1) / effective_step))

        while chunk_start < total_duration:
            chunk_end = min(chunk_start + CHUNK_DURATION, total_duration)
            chunk_num += 1

            # Get segments for this chunk
            chunk_segments = [
                s for s in segments
                if s.get('end', 0) > chunk_start and s.get('start', 0) < chunk_end
            ]

            if not chunk_segments:
                chunk_start = chunk_end
                continue

            condensed = self._condense_transcript(chunk_segments)

            print(
                f"  Chunk {chunk_num}/{total_chunks}: "
                f"{self._format_ts(chunk_start)} - {self._format_ts(chunk_end)} "
                f"({len(chunk_segments)} segments, {len(condensed)} chars)"
            )

            try:
                prompt = self._build_chunk_prompt(
                    condensed, chunk_start, chunk_end, total_duration
                )
                response = self._call_llm(prompt)

                # Parse chapters — use chunk_end as the duration boundary
                chunk_chapters = self._parse_response(
                    response, chunk_segments, chunk_end
                )

                # Filter out chapters that start before this chunk
                # (can happen due to overlap context)
                if all_chapters:
                    prev_end = all_chapters[-1].get('end_time', chunk_start)
                    chunk_chapters = [
                        c for c in chunk_chapters
                        if c['start_time'] >= prev_end - OVERLAP / 2
                    ]

                    # Adjust previous chapter's end time to connect with new chapters
                    if chunk_chapters and all_chapters:
                        all_chapters[-1]['end_time'] = chunk_chapters[0]['start_time']

                all_chapters.extend(chunk_chapters)

            except Exception as e:
                logger.warning(f"Chunk {chunk_num} failed: {e}")
                print(f"  Warning: chunk {chunk_num} failed ({e}), skipping")

            # Move to next chunk
            if chunk_end >= total_duration:
                break  # Covered the entire video
            chunk_start = chunk_end - OVERLAP

        if not all_chapters:
            return self._fallback_chapters(segments, total_duration)

        # Ensure last chapter covers to the end
        if all_chapters:
            all_chapters[-1]['end_time'] = total_duration

        # Re-index and attach transcript segments
        for i, ch in enumerate(all_chapters):
            ch['index'] = i + 1
            ch['transcript_segments'] = [
                seg for seg in segments
                if seg.get('end', 0) > ch['start_time'] and seg.get('start', 0) < ch['end_time']
            ]

        return all_chapters

    def _fallback_chapters(
        self, segments: List[Dict], total_duration: float
    ) -> List[Dict]:
        """Fallback: divide into equal-duration chapters."""
        num = max(1, round(total_duration / self.target_duration))
        duration_each = total_duration / num

        chapters = []
        for i in range(num):
            start = i * duration_each
            end = min((i + 1) * duration_each, total_duration)
            ch_segments = [
                seg for seg in segments
                if seg.get('end', 0) > start and seg.get('start', 0) < end
            ]
            chapters.append({
                'index': i + 1,
                'start_time': start,
                'end_time': end,
                'title': f'段落 {i + 1}',
                'summary': '',
                'highlight': 3,
                'tags': [],
                'transcript_segments': ch_segments,
            })
        return chapters

    def _group_chapters(
        self, sub_chapters: List[Dict], total_duration: float
    ) -> List[Dict]:
        """Group sub-chapters into major chapters using LLM.

        Takes a flat list of sub-chapters and asks the LLM to group them
        into a smaller number of major chapters (sections).
        """
        if len(sub_chapters) <= 4:
            # Too few sub-chapters to group meaningfully
            highlight = max((sc.get('highlight', 3) for sc in sub_chapters), default=3)
            all_tags = []
            for sc in sub_chapters:
                for tag in sc.get('tags', []):
                    if tag not in all_tags:
                        all_tags.append(tag)
            return [{
                'index': 1,
                'title': sub_chapters[0]['title'] if sub_chapters else '全部內容',
                'start_time': sub_chapters[0]['start_time'] if sub_chapters else 0,
                'end_time': total_duration,
                'summary': '',
                'highlight': highlight,
                'tags': all_tags,
                'sub_chapters': sub_chapters,
            }]

        # Build a compact list of sub-chapters for the LLM
        lines = []
        for ch in sub_chapters:
            ts = self._format_ts(ch['start_time'])
            dur = (ch['end_time'] - ch['start_time']) / 60
            stars = '★' * ch.get('highlight', 3) + '☆' * (5 - ch.get('highlight', 3))
            lines.append(f"{ch['index']}. [{ts}] ({dur:.0f}m) {stars} {ch['title']} — {ch.get('summary', '')}")
        chapter_list = '\n'.join(lines)

        num_major = max(2, min(8, len(sub_chapters) // 4))

        prompt = f"""你是影片章節分析助手。以下是一部影片的細分章節列表，請將它們分組成幾個大的主要章節。

**影片資訊：**
- 總長度：{self._format_ts(total_duration)}
- 共 {len(sub_chapters)} 個小章節
- 建議分成約 {num_major} 個大章節（可依內容調整）

**小章節列表：**
{chapter_list}

**任務：**
將以上小章節按照內容主題分組成大章節。每個大章節應該包含多個相關的小章節。

**要求：**
1. 每個大章節用 from/to 指定包含的小章節編號範圍
2. 大章節標題要能概括該組小章節的整體主題（5-15 字）
3. 摘要用一到兩句話描述該大章節的整體內容
4. 所有小章節都必須被包含，不能遺漏

**語言：請全程使用繁體中文（zh-TW）輸出。**

**請只輸出以下 JSON 格式，不要輸出任何其他文字：**
{{"groups": [{{"from": 1, "to": 6, "title": "大章節標題", "summary": "該大章節摘要"}}]}}"""

        try:
            response = self._call_llm(prompt)
            major_chapters = self._parse_groups_response(response, sub_chapters, total_duration)
        except Exception as e:
            logger.warning(f"Chapter grouping failed: {e}")
            print(f"  Grouping failed ({e}), using flat structure")
            major_chapters = self._fallback_grouping(sub_chapters, total_duration)

        # Aggregate highlight/tags from sub-chapters
        for mc in major_chapters:
            subs = mc.get('sub_chapters', [])
            if subs:
                mc['highlight'] = max(sc.get('highlight', 3) for sc in subs)
                all_tags = []
                for sc in subs:
                    for tag in sc.get('tags', []):
                        if tag not in all_tags:
                            all_tags.append(tag)
                mc['tags'] = all_tags
            else:
                mc['highlight'] = 3
                mc['tags'] = []

        return major_chapters

    def _parse_groups_response(
        self, response: str, sub_chapters: List[Dict], total_duration: float
    ) -> List[Dict]:
        """Parse LLM grouping response into major chapter structure."""
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        json_str = None
        m = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if m:
                json_str = m.group(0)

        if not json_str:
            raise ValueError(f"No JSON in grouping response:\n{response[:500]}")

        data = json.loads(json_str)
        raw_groups = data.get('groups', [])
        if not raw_groups:
            raise ValueError("Empty groups in response")

        major_chapters = []
        used_indices = set()

        for i, g in enumerate(raw_groups):
            from_idx = int(g.get('from', 1))
            to_idx = int(g.get('to', len(sub_chapters)))
            title = str(g.get('title', f'段落 {i + 1}')).strip()
            summary = str(g.get('summary', '')).strip()

            # Collect sub-chapters in this range
            subs = [ch for ch in sub_chapters if from_idx <= ch['index'] <= to_idx]
            if not subs:
                continue

            for ch in subs:
                used_indices.add(ch['index'])

            major_chapters.append({
                'index': i + 1,
                'title': title,
                'start_time': subs[0]['start_time'],
                'end_time': subs[-1]['end_time'],
                'summary': summary,
                'sub_chapters': subs,
            })

        # Check for any unassigned sub-chapters
        unassigned = [ch for ch in sub_chapters if ch['index'] not in used_indices]
        if unassigned:
            if major_chapters:
                major_chapters[-1]['sub_chapters'].extend(unassigned)
                major_chapters[-1]['end_time'] = max(
                    major_chapters[-1]['end_time'],
                    unassigned[-1]['end_time']
                )
            else:
                major_chapters.append({
                    'index': 1,
                    'title': '其他',
                    'start_time': unassigned[0]['start_time'],
                    'end_time': unassigned[-1]['end_time'],
                    'summary': '',
                    'sub_chapters': unassigned,
                })

        # Re-index
        for i, mc in enumerate(major_chapters):
            mc['index'] = i + 1

        return major_chapters

    def _fallback_grouping(
        self, sub_chapters: List[Dict], total_duration: float
    ) -> List[Dict]:
        """Fallback: group sub-chapters into roughly equal major chapters."""
        group_size = max(3, len(sub_chapters) // 5)
        major_chapters = []

        for i in range(0, len(sub_chapters), group_size):
            subs = sub_chapters[i:i + group_size]
            major_chapters.append({
                'index': len(major_chapters) + 1,
                'title': subs[0]['title'],
                'start_time': subs[0]['start_time'],
                'end_time': subs[-1]['end_time'],
                'summary': '',
                'sub_chapters': subs,
            })

        return major_chapters

    def detect_chapters(
        self, segments: List[Dict], total_duration: float
    ) -> List[Dict]:
        """
        Detect chapter boundaries with hierarchical structure.

        Returns major chapters, each containing sub-chapters.
        Automatically uses chunked detection for videos > 30 minutes.

        Args:
            segments: List of segment dicts with 'start', 'end', 'text' keys
            total_duration: Total video/audio duration in seconds

        Returns:
            List of major chapter dicts with keys:
            - index, start_time, end_time, title, summary, sub_chapters
            Each sub_chapter has: index, start_time, end_time, title, summary, transcript_segments
        """
        if not segments:
            return []

        CHUNK_THRESHOLD = 1800  # Use chunked detection for videos > 30 min

        print(f"\n--- Chapter Detection ---")
        print(f"Model: {self.model}")
        print(f"Segments: {len(segments)}")
        print(f"Duration: {self._format_ts(total_duration)}")

        # Step 1: Detect fine-grained sub-chapters
        try:
            if total_duration > CHUNK_THRESHOLD:
                print(f"Using chunked detection (video > {CHUNK_THRESHOLD // 60} min)...")
                sub_chapters = self._detect_chunked(segments, total_duration)
            else:
                condensed = self._condense_transcript(segments)
                print(f"Condensed: {len(condensed)} chars")
                prompt = self._build_prompt(condensed, total_duration)
                response = self._call_llm(prompt)
                sub_chapters = self._parse_response(response, segments, total_duration)
        except Exception as e:
            logger.warning(f"Chapter detection failed: {e}")
            print(f"Chapter detection failed: {e}")
            print("Falling back to time-based division...")
            sub_chapters = self._fallback_chapters(segments, total_duration)

        print(f"\nDetected {len(sub_chapters)} sub-chapters")

        # Step 2: Group sub-chapters into major chapters
        if len(sub_chapters) > 4:
            print(f"Grouping into major chapters...")
            major_chapters = self._group_chapters(sub_chapters, total_duration)
        else:
            highlight = max((sc.get('highlight', 3) for sc in sub_chapters), default=3)
            all_tags = []
            for sc in sub_chapters:
                for tag in sc.get('tags', []):
                    if tag not in all_tags:
                        all_tags.append(tag)
            major_chapters = [{
                'index': 1,
                'title': '全部內容',
                'start_time': sub_chapters[0]['start_time'] if sub_chapters else 0,
                'end_time': total_duration,
                'summary': '',
                'highlight': highlight,
                'tags': all_tags,
                'sub_chapters': sub_chapters,
            }]

        # Print results
        print(f"\n{'='*50}")
        print(f"Chapters: {len(major_chapters)} major, {len(sub_chapters)} sub-chapters")
        print(f"{'='*50}")
        for mc in major_chapters:
            dur = (mc['end_time'] - mc['start_time']) / 60
            mc_stars = '★' * mc.get('highlight', 3) + '☆' * (5 - mc.get('highlight', 3))
            print(
                f"\n{mc['index']}. {mc_stars} {mc['title']} "
                f"[{self._format_ts(mc['start_time'])} - {self._format_ts(mc['end_time'])}] "
                f"({dur:.0f}m)"
            )
            for sc in mc['sub_chapters']:
                dur_sc = (sc['end_time'] - sc['start_time']) / 60
                sc_stars = '★' * sc.get('highlight', 3) + '☆' * (5 - sc.get('highlight', 3))
                print(
                    f"   {sc['index']:2d}. [{self._format_ts(sc['start_time'])}] "
                    f"({dur_sc:.0f}m) {sc_stars} {sc['title']}"
                )

        return major_chapters

    @staticmethod
    def save_chapters_md(
        major_chapters: List[Dict],
        output_path,
        video_name: str = "",
        total_duration: float = 0,
    ):
        """Save hierarchical chapters as Markdown."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _ts = ChapterSegmenter._format_ts
        _tsp = ChapterSegmenter._format_ts_padded

        total_subs = sum(len(mc.get('sub_chapters', [])) for mc in major_chapters)

        lines = []

        # Header
        if video_name:
            lines.append(f"# {video_name} - 章節列表")
        else:
            lines.append("# 章節列表")

        if total_duration:
            h = int(total_duration // 3600)
            m = int((total_duration % 3600) // 60)
            s = int(total_duration % 60)
            lines.append(f"# 影片長度：{h}h {m:02d}m {s:02d}s")

        lines.append(f"# 共 {len(major_chapters)} 個大章節，{total_subs} 個小章節")
        lines.append("")

        # YouTube-style chapter markers (copy-paste ready)
        lines.append("## 時間軸")
        lines.append("")
        for mc in major_chapters:
            dur = (mc['end_time'] - mc['start_time']) / 60
            mc_stars = '★' * mc.get('highlight', 3) + '☆' * (5 - mc.get('highlight', 3))
            lines.append(f"# {mc_stars} {mc['title']} ({dur:.0f}m)")
            for sc in mc.get('sub_chapters', []):
                ts = _tsp(sc['start_time'])
                sc_stars = '★' * sc.get('highlight', 3) + '☆' * (5 - sc.get('highlight', 3))
                tag_str = ' '.join(f"#{t}" for t in sc.get('tags', []))
                suffix = f" {tag_str}" if tag_str else ""
                lines.append(f"{ts} {sc_stars} {sc['title']}{suffix}")
            lines.append("")

        # Detailed chapter info
        lines.append("## 章節詳情")
        lines.append("")
        for mc in major_chapters:
            dur = (mc['end_time'] - mc['start_time']) / 60
            mc_stars = '★' * mc.get('highlight', 3) + '☆' * (5 - mc.get('highlight', 3))
            lines.append(f"### {mc['index']}. {mc_stars} {mc['title']}")
            lines.append(f"時間：{_ts(mc['start_time'])} - {_ts(mc['end_time'])} ({dur:.0f} 分鐘)")
            if mc.get('summary'):
                lines.append(f"摘要：{mc['summary']}")
            mc_tags = mc.get('tags', [])
            if mc_tags:
                lines.append(f"標籤：{' '.join(f'#{t}' for t in mc_tags)}")
            lines.append("")

            for sc in mc.get('sub_chapters', []):
                sc_dur = (sc['end_time'] - sc['start_time']) / 60
                sc_stars = '★' * sc.get('highlight', 3) + '☆' * (5 - sc.get('highlight', 3))
                lines.append(f"#### {mc['index']}.{sc['index']}. {sc_stars} {sc['title']}")
                lines.append(f"時間：{_ts(sc['start_time'])} - {_ts(sc['end_time'])} ({sc_dur:.0f} 分鐘)")
                if sc.get('summary'):
                    lines.append(f"摘要：{sc['summary']}")
                sc_tags = sc.get('tags', [])
                if sc_tags:
                    lines.append(f"標籤：{' '.join(f'#{t}' for t in sc_tags)}")
                lines.append("")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"Chapters markdown saved to: {output_path}")

    @staticmethod
    def save_chapters_json(
        major_chapters: List[Dict],
        output_path,
        video_name: str = "",
        total_duration: float = 0,
    ):
        """Save hierarchical chapters in JSON format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _ts = ChapterSegmenter._format_ts

        total_subs = sum(len(mc.get('sub_chapters', [])) for mc in major_chapters)

        data = {
            'video_name': video_name,
            'total_duration': total_duration,
            'num_major_chapters': len(major_chapters),
            'num_sub_chapters': total_subs,
            'chapters': [
                {
                    'index': mc['index'],
                    'title': mc['title'],
                    'summary': mc.get('summary', ''),
                    'highlight': mc.get('highlight', 3),
                    'tags': mc.get('tags', []),
                    'start_time': mc['start_time'],
                    'end_time': mc['end_time'],
                    'start_formatted': _ts(mc['start_time']),
                    'end_formatted': _ts(mc['end_time']),
                    'sub_chapters': [
                        {
                            'index': sc['index'],
                            'title': sc['title'],
                            'summary': sc.get('summary', ''),
                            'highlight': sc.get('highlight', 3),
                            'tags': sc.get('tags', []),
                            'start_time': sc['start_time'],
                            'end_time': sc['end_time'],
                            'start_formatted': _ts(sc['start_time']),
                            'end_formatted': _ts(sc['end_time']),
                        }
                        for sc in mc.get('sub_chapters', [])
                    ],
                }
                for mc in major_chapters
            ],
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Chapters JSON saved to: {output_path}")

    @staticmethod
    def _format_tc(seconds: float, fps: int = 24) -> str:
        """Format seconds as HH:MM:SS:FF (frame-based timecode)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        f = int((seconds % 1) * fps)
        return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"

    @staticmethod
    def save_chapters_csv(
        major_chapters: List[Dict],
        output_path,
        video_name: str = "",
        total_duration: float = 0,
        fps: int = 24,
    ):
        """Save chapters as DaVinci Resolve marker CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _tc = ChapterSegmenter._format_tc

        rows = []
        for mc in major_chapters:
            color = 'Red' if mc.get('highlight', 3) >= 4 else 'Blue'
            start_tc = _tc(mc['start_time'], fps)
            end_tc = _tc(mc['end_time'], fps)
            dur_tc = _tc(mc['end_time'] - mc['start_time'], fps)
            rows.append({
                'Name': f"{mc['index']}. {mc['title']}",
                'Start TC': start_tc,
                'End TC': end_tc,
                'Duration': dur_tc,
                'Color': color,
                'Notes': mc.get('summary', ''),
            })
            for sc in mc.get('sub_chapters', []):
                sc_color = 'Red' if sc.get('highlight', 3) >= 4 else 'Green'
                sc_start = _tc(sc['start_time'], fps)
                sc_end = _tc(sc['end_time'], fps)
                sc_dur = _tc(sc['end_time'] - sc['start_time'], fps)
                tags_str = ', '.join(sc.get('tags', []))
                notes = sc.get('summary', '')
                if tags_str:
                    notes = f"[{tags_str}] {notes}" if notes else f"[{tags_str}]"
                rows.append({
                    'Name': f"  {mc['index']}.{sc['index']} {sc['title']}",
                    'Start TC': sc_start,
                    'End TC': sc_end,
                    'Duration': sc_dur,
                    'Color': sc_color,
                    'Notes': notes,
                })

        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Name', 'Start TC', 'End TC', 'Duration', 'Color', 'Notes'])
            writer.writeheader()
            writer.writerows(rows)

        print(f"Chapters CSV saved to: {output_path}")

    @staticmethod
    def save_chapters_edl(
        major_chapters: List[Dict],
        output_path,
        video_name: str = "",
        total_duration: float = 0,
        fps: int = 24,
    ):
        """Save chapters as CMX 3600 EDL for NLE import."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _tc = ChapterSegmenter._format_tc

        lines = []
        lines.append(f"TITLE: {video_name or 'Untitled'}")
        lines.append("FCM: NON-DROP FRAME")
        lines.append("")

        event_num = 1
        for mc in major_chapters:
            for sc in mc.get('sub_chapters', []):
                src_in = _tc(sc['start_time'], fps)
                src_out = _tc(sc['end_time'], fps)
                rec_in = _tc(sc['start_time'], fps)
                rec_out = _tc(sc['end_time'], fps)
                lines.append(
                    f"{event_num:03d}  AX       V     C        "
                    f"{src_in} {src_out} {rec_in} {rec_out}"
                )
                lines.append(f"* FROM CLIP NAME: {sc['title']}")
                if sc.get('summary'):
                    lines.append(f"* COMMENT: {sc['summary']}")
                lines.append("")
                event_num += 1

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"Chapters EDL saved to: {output_path}")
