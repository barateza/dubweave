import os
import time
import shutil
from pathlib import Path
from typing import cast
import gradio as gr

from src.config import (
    WORK_DIR, WHISPER_MODEL, KOKORO_VOICE, GOOGLE_TTS_API_KEY, 
    GOOGLE_TTS_LANGUAGE_CODE, GOOGLE_TTS_VOICE_TYPE, GOOGLE_TTS_VOICE_NAME,
    EDGE_TTS_VOICE_NAME
)
from src.utils.helpers import log
from src.utils.security import validate_openrouter_key, validate_google_tts_key
from src.utils.system import release_gpu_memory
from src.utils.project import (
    project_dir, save_project_stage, load_project_stage, 
    cleanup_stale_jobs, generate_srt_for_project
)
from src.core.translate import (
    PipelineError, translate_segments, get_merge_config, 
    group_for_synthesis as _group_for_synthesis
)
from src.core.ingest import validate_video_source, ingest_local_file, download_video
from src.core.transcribe import transcribe_audio
from src.core.synthesis import (
    apply_timing_budget, get_cps_for_voice, synthesize_segments_kokoro,
    synthesize_segments_google_tts, synthesize_segments_edge_tts,
    synthesize_segments, assemble_dubbed_video
)

STAGES = ["download", "transcribe", "translate", "synthesize", "assemble"]

def lazy_import():
    global yt_dlp, whisper, torch, TTS
    import yt_dlp
    import whisper
    import torch
    from TTS.api import TTS
    return True

def run_pipeline(
    url: str,
    video_upload_path: str | None,
    speaker_wav_path: str | None,
    whisper_model: str,
    browser: str,
    cookies_file: str | None,
    project_name: str,
    resume_from: str,
    tts_engine: str = "XTTS v2 (voice clone)",
    kokoro_voice: str = KOKORO_VOICE,
    google_tts_voice_type: str = GOOGLE_TTS_VOICE_TYPE,
    google_tts_voice_name: str = GOOGLE_TTS_VOICE_NAME,
    edge_tts_voice: str = EDGE_TTS_VOICE_NAME,
    progress=gr.Progress(),
):
    logs = []
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    proj = project_name.strip() or "default"
    
    stage_order = {s: i for i, s in enumerate(STAGES)}
    resume_idx = stage_order.get(resume_from, 0)

    job_id = str(int(time.time()))
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    model_to_use = whisper_model.strip() if whisper_model.strip() else WHISPER_MODEL

    try:
        lazy_import()
        logs = cleanup_stale_jobs(logs)
        logs = log(f"📁 Project: {proj}  |  Resume from: {resume_from}", logs)
        yield None, "\n".join(logs)

        # ── Pre-flight validation ───────────────────────────────────────────
        if resume_idx <= stage_order["download"]:
            src_ok, src_result = validate_video_source(url, video_upload_path)
            if not src_ok: raise PipelineError("Validation", src_result)
            source_mode = src_result

        if openrouter_key:
            logs = log("🔑 Validating OpenRouter API key…", logs); yield None, "\n".join(logs)
            ok, msg = validate_openrouter_key(openrouter_key)
            if not ok: raise PipelineError("Validation", f"OpenRouter key invalid: {msg}")
            log("   ✅ OpenRouter key valid", logs); yield None, "\n".join(logs)

        if tts_engine == "Google Cloud TTS":
            if not GOOGLE_TTS_API_KEY: raise PipelineError("Validation", "GOOGLE_TTS_API_KEY missing.")
            logs = log("🔑 Validating Google TTS API key…", logs); yield None, "\n".join(logs)
            ok, msg = validate_google_tts_key(GOOGLE_TTS_API_KEY)
            if not ok: raise PipelineError("Validation", f"Google TTS key invalid: {msg}")
            log("   ✅ Google TTS key valid", logs); yield None, "\n".join(logs)

        # ── Download / Ingest ─────────────────────────────────────────────────
        if resume_idx <= stage_order["download"]:
            progress(0.05, desc="Downloading/Ingesting…")
            if source_mode == "file":
                video_path, audio_path, title, duration, logs = ingest_local_file(video_upload_path, job_dir, logs)
            else:
                video_path, audio_path, title, duration, logs = download_video(url, job_dir, logs, browser=browser, cookies_file=cookies_file)
            save_project_stage(proj, "download", (video_path, audio_path, title, duration))
            yield None, "\n".join(logs)
        else:
            log("⏭️  Skipping download (loaded from project)", logs)
            video_path, audio_path, title, duration = load_project_stage(proj, "download")
            yield None, "\n".join(logs)

        # ── Transcribe ────────────────────────────────────────────────────────
        if resume_idx <= stage_order["transcribe"]:
            progress(0.2, desc="Transcribing…")
            segments, logs = transcribe_audio(audio_path, logs, model_name=model_to_use)
            save_project_stage(proj, "transcribe", segments)
            yield None, "\n".join(logs)
            release_gpu_memory()
        else:
            segments = load_project_stage(proj, "transcribe")
            log(f"⏭️  Skipping transcription ({len(segments)} segments loaded)", logs); yield None, "\n".join(logs)

        # ── Translate ─────────────────────────────────────────────────────────
        if resume_idx <= stage_order["translate"]:
            progress(0.4, desc="Translating…")
            m_cfg = get_merge_config(tts_engine)
            translated, logs = translate_segments(segments, logs, openrouter_key=openrouter_key, merge_config=m_cfg)
            
            progress(0.5, desc="Checking timing budget…")
            active_voice = kokoro_voice if tts_engine.startswith("Kokoro") else (edge_tts_voice if tts_engine.startswith("Edge") else (google_tts_voice_name if tts_engine.startswith("Google") else "default"))
            cps = get_cps_for_voice(tts_engine, active_voice)
            translated, logs = apply_timing_budget(translated, logs, openrouter_key=openrouter_key, cps=cps)
            
            save_project_stage(proj, "translate", translated)
            yield None, "\n".join(logs)
            release_gpu_memory()
        else:
            translated = load_project_stage(proj, "translate")
            log(f"⏭️  Skipping translation ({len(translated)} segments loaded)", logs); yield None, "\n".join(logs)

        # ── Synthesize ────────────────────────────────────────────────────────
        if resume_idx <= stage_order["synthesize"]:
            progress(0.55, desc="Synthesizing voice…")
            utterances = _group_for_synthesis(translated)
            if tts_engine == "Kokoro (fast, PT-BR native)":
                timed_clips, logs = synthesize_segments_kokoro(utterances, job_dir, logs, voice=kokoro_voice)
            elif tts_engine == "Google Cloud TTS":
                timed_clips, logs = synthesize_segments_google_tts(utterances, job_dir, logs, api_key=GOOGLE_TTS_API_KEY, voice_type=google_tts_voice_type, voice_name=google_tts_voice_name, language_code=GOOGLE_TTS_LANGUAGE_CODE)
            elif tts_engine == "Edge TTS (cloud, no key)":
                timed_clips, logs = synthesize_segments_edge_tts(utterances, job_dir, logs, voice=edge_tts_voice)
            else:
                timed_clips, logs = synthesize_segments(utterances, audio_path, job_dir, logs, speaker_wav=speaker_wav_path)
            
            save_project_stage(proj, "synthesize", timed_clips)
            yield None, "\n".join(logs)
            release_gpu_memory()
        else:
            timed_clips = load_project_stage(proj, "synthesize")
            log(f"⏭️  Skipping synthesis ({len(timed_clips)} clips loaded)", logs); yield None, "\n".join(logs)

        # ── Assemble ──────────────────────────────────────────────────────────
        progress(0.85, desc="Assembling video…")
        output_path, logs = assemble_dubbed_video(video_path, timed_clips, float(duration or 0), job_dir, title or "video", logs)
        save_project_stage(proj, "assemble", output_path)

        try:
            _, srt_msg = generate_srt_for_project(proj)
            log(f"📄 {srt_msg}", logs)
        except Exception as e: log(f"⚠️  SRT failed: {e}", logs)

        progress(1.0, desc="Done!")
        yield output_path, "\n".join(logs)

    except PipelineError as e:
        log(f"❌ [{e.stage}] {e.message}", logs); yield None, "\n".join(logs)
    except Exception as e:
        import traceback
        log(f"❌ Unexpected error: {e}\n{traceback.format_exc()}", logs); yield None, "\n".join(logs)
    finally:
        shutil.rmtree(str(job_dir), ignore_errors=True)
