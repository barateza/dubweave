---
title: Dubweave Demo
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.16.0"
python_version: "3.11"
app_file: app.py
pinned: false
short_description: AI dubbing demo — transcribe, translate, re-voice any video
---

# Dubweave Demo

This Hugging Face Space runs a lightweight, CPU-optimized build of **Dubweave**. 

## Demo features & limitations:

- **File Upload Only**: Designed for quick testing of local video files (`.mp4`, `.mkv`, etc.).
- **Edge TTS (Speech Synthesis)**: Powered entirely by free, highly realistic Neural Edge TTS voices running locally on CPU.
- **Dynamic Voice Customization**: Allows choosing standard male/female neural voices for each target language.
- **Smart Translation Routing**:
  - **English → Portuguese (BR)**: Works entirely offline/locally for free using a local `Helsinki-NLP` translation model (no API key required).
  - **All other language pairs**: Route through OpenRouter using high-performance `Gemini 2.0 Flash` (requires providing an OpenRouter API key).
  - **Same language pairs**: Skips translation entirely (useful for re-voicing/accent adjustment).
- **Dual Downloader outputs**: Dubbed video (`dubbed_video.mp4`) and translated subtitles (`subtitles.srt`) are immediately generated as separate file downloads.
