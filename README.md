# LongVideoHelper

This project aim to offer a tool that use whisper and VLM for long video chapter dividing.

## Workflow

User input a Video, and we will create the pure audio (mp4). Since the whisper can't process the long audio well, so we first use the volume to divided the video into several clips, all of them are shorter than 5 mins.

While processing the audio, since the video may up to 3 hours, the clip algorithm is we will first clip the video with webrtcvad, make sure each clip will not cut one sentence, than I want each clip is smaller than 5 minutes because whisper will perform more accuracy in this length. Once we get a clip, we will use the whisper to transcribe.

After whole video finish the transcribe, than we will try to ask VLM to seperate the video into several chapter. Since the transcribe result may not be correct, so we need the aid of the video. First we will send the whole transcript, and ask the VLM to output the detailed chapter start and end time, each chapter should control in 6 minutes. Then for each chapter, we will get the key frames with transcribe, and ask LLM to correct the transcribe while output the summary for this chapter.

We expected output is a full-video transcript and the summary md for each chapter information.

## Technuqie stack

- Whisper turbo
- VLM:
  - Cloud: Gemini flash 2.5
  - Local: Gemma3-4B or Gemma3-12B, Qwen3-VL-8B
  - LLMInterface: create a llm.py, and use the LiteLLM to allow using cloud or ollama etc.
- Interface: CLI, ready for GUI(TKinter in future)
- Use uv.
