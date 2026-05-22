---
title: Dubweave Demo
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
short_description: AI video dubbing demo for CPU-only Hugging Face Spaces
---

# Dubweave Demo

This Space is the lightweight demo build of Dubweave.

## Demo scope

- File upload only
- Input and output language selectors
- Edge TTS voice gender and voice selection
- Dubbed video output plus separate SRT subtitles

## Notes

- English to Portuguese (BR) can use the local fallback path.
- Other language pairs should provide an OpenRouter API key.
- The demo is tuned for CPU-only Spaces, so the first run may still take a little while.
