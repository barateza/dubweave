import os
import time
import shutil
from pathlib import Path
from typing import cast
import gradio as gr

from src.config import (
    WORK_DIR,
    WHISPER_MODEL,
    DEFAULT_INPUT_LANGUAGE,
    DEFAULT_OUTPUT_LANGUAGE,
    INPUT_LANGUAGE_OPTIONS,
    EDGE_TTS_VOICE_NAME,
    OPENROUTER_API_KEY,
    OUTPUT_DIR,
)
from src.utils.helpers import log
from src.utils.security import validate_openrouter_key
from src.utils.system import release_gpu_memory
from src.utils.project import (
    project_dir,
    save_project_stage,
    load_project_stage,
    cleanup_stale_jobs,
    generate_srt_for_project,
)
from src.core.translate import (
    PipelineError,
    translate_segments,
    get_merge_config,
    group_for_synthesis as _group_for_synthesis,
)
from src.core.ingest import validate_video_source, ingest_local_file, download_video
from src.core.transcribe import transcribe_audio, format_tqdm
from src.core.synthesis import (
    apply_timing_budget,
    get_cps_for_voice,
    synthesize_segments_edge_tts,
    assemble_dubbed_video,
)

STAGES = ["download", "transcribe", "translate", "synthesize", "assemble"]


def lazy_import():
    global yt_dlp, torch
    try:
        import yt_dlp
    except ImportError:
        yt_dlp = None
    import torch
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
    tts_engine: str = "Edge TTS (cloud, no key)",
    kokoro_voice: str = "",
    google_tts_voice_type: str = "",
    google_tts_voice_name: str = "",
    edge_tts_voice: str = EDGE_TTS_VOICE_NAME,
    supertonic_lang: str = "",
    supertonic_voice: str = "",
    gemini_pricing_mode: str = "auto",
    gemini_single_voice: str = "",
    gemini_multi_speaker: bool = False,
    gemini_speaker_assignment: str = "",
    gemini_speaker1_name: str = "",
    gemini_speaker1_voice: str = "",
    gemini_speaker2_name: str = "",
    gemini_speaker2_voice: str = "",
    elevenlabs_voice_id: str = "",
    elevenlabs_model_id: str = "",
    input_language: str = DEFAULT_INPUT_LANGUAGE,
    output_language: str = DEFAULT_OUTPUT_LANGUAGE,
    openrouter_api_key: str = "",
    progress=gr.Progress(),
):
    logs = []
    proj = project_name.strip() or "default"

    stage_order = {s: i for i, s in enumerate(STAGES)}
    resume_idx = stage_order.get(resume_from, 0)

    job_id = str(int(time.time()))
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    model_to_use = whisper_model.strip() if whisper_model.strip() else WHISPER_MODEL
    input_language_value = (input_language or DEFAULT_INPUT_LANGUAGE).strip()
    input_language_code = INPUT_LANGUAGE_OPTIONS.get(input_language_value, input_language_value)
    if input_language_code == "":
        input_language_code = None
    output_language_value = (output_language or DEFAULT_OUTPUT_LANGUAGE).strip() or DEFAULT_OUTPUT_LANGUAGE
    openrouter_key = openrouter_api_key.strip() or OPENROUTER_API_KEY

    try:
        lazy_import()
        logs = cleanup_stale_jobs(logs)
        logs = log(f"📁 Project: {proj}  |  Resume from: {resume_from}", logs)
        yield None, None, None, "\n".join(logs)

        # ── Pre-flight validation ───────────────────────────────────────────
        if resume_idx <= stage_order["download"]:
            src_ok, src_result = validate_video_source(url, video_upload_path)
            if not src_ok:
                raise PipelineError("Validation", src_result)
            source_mode = src_result

        if openrouter_key:
            logs = log("🔑 Validating OpenRouter API key…", logs)
            yield None, None, None, "\n".join(logs)
            ok, msg = validate_openrouter_key(openrouter_key)
            if not ok:
                raise PipelineError("Validation", f"OpenRouter key invalid: {msg}")
            log("   ✅ OpenRouter key valid", logs)
            yield None, None, None, "\n".join(logs)

        # ── Download / Ingest ─────────────────────────────────────────────────
        if resume_idx <= stage_order["download"]:
            progress(0.05, desc="Downloading/Ingesting…")
            if source_mode == "file":
                video_path, audio_path, title, duration, logs = ingest_local_file(
                    video_upload_path, job_dir, logs
                )
            else:
                video_path, audio_path, title, duration, logs = download_video(
                    url, job_dir, logs, browser=browser, cookies_file=cookies_file
                )
            save_project_stage(
                proj, "download", (video_path, audio_path, title, duration)
            )
            yield None, None, None, "\n".join(logs)
        else:
            log("⏭️  Skipping download (loaded from project)", logs)
            video_path, audio_path, title, duration = load_project_stage(
                proj, "download"
            )
            yield None, None, None, "\n".join(logs)

        # ── Transcribe ────────────────────────────────────────────────────────
        if resume_idx <= stage_order["transcribe"]:
            progress(0.2, desc="Transcribing…")
            logs = log(f"🎙️ Transcribing with Whisper ({model_to_use})…", logs)
            if input_language_code:
                logs = log(f"   Using language hint: {input_language_code}", logs)
            else:
                logs = log("   Auto-detecting language…", logs)
            yield None, None, None, "\n".join(logs)

            last_progress = ""
            segments, detected_lang = None, None

            for event_type, data in transcribe_audio(
                audio_path,
                logs,
                model_name=model_to_use,
                language=input_language_code,
            ):
                if event_type == "log":
                    formatted = format_tqdm(data)
                    if formatted:
                        if formatted.startswith("   ⏳"):
                            if last_progress and last_progress in logs:
                                logs.remove(last_progress)
                            last_progress = formatted
                            logs.append(formatted)
                        else:
                            logs = log(formatted, logs)
                        yield None, None, None, "\n".join(logs)
                elif event_type == "done":
                    segments, detected_lang = data

            logs = log(f"   Detected language: {detected_lang}", logs)
            logs = log(f"✅ Transcribed {len(segments)} segments", logs)
            save_project_stage(proj, "transcribe", segments)
            yield None, None, None, "\n".join(logs)
            release_gpu_memory()
        else:
            segments = load_project_stage(proj, "transcribe")
            detected_lang = input_language_code or "unknown"
            log(f"⏭️  Skipping transcription ({len(segments)} segments loaded)", logs)
            yield None, None, None, "\n".join(logs)

        # ── Translate ─────────────────────────────────────────────────────────
        if resume_idx <= stage_order["translate"]:
            progress(0.4, desc="Translating…")
            m_cfg = get_merge_config("edge")
            translated, logs = translate_segments(
                segments,
                logs,
                openrouter_key=openrouter_key,
                merge_config=m_cfg,
                source_lang=detected_lang,
                target_language=output_language_value,
            )

            progress(0.5, desc="Checking timing budget…")
            cps = get_cps_for_voice("edge", edge_tts_voice)
            translated, logs = apply_timing_budget(
                translated, logs, openrouter_key=openrouter_key, cps=cps
            )

            save_project_stage(proj, "translate", translated)
            yield None, None, None, "\n".join(logs)
            release_gpu_memory()
        else:
            translated = load_project_stage(proj, "translate")
            log(f"⏭️  Skipping translation ({len(translated)} segments loaded)", logs)
            yield None, None, None, "\n".join(logs)

        # ── Synthesize ────────────────────────────────────────────────────────
        if resume_idx <= stage_order["synthesize"]:
            progress(0.55, desc="Synthesizing voice…")
            utterances = _group_for_synthesis(translated)
            timed_clips, logs = synthesize_segments_edge_tts(
                utterances, job_dir, logs, voice=edge_tts_voice
            )

            save_project_stage(proj, "synthesize", timed_clips)
            yield None, None, None, "\n".join(logs)
            release_gpu_memory()
        else:
            timed_clips = load_project_stage(proj, "synthesize")
            log(f"⏭️  Skipping synthesis ({len(timed_clips)} clips loaded)", logs)
            yield None, None, None, "\n".join(logs)

        # ── Assemble ──────────────────────────────────────────────────────────
        progress(0.85, desc="Assembling video…")
        output_path, logs = assemble_dubbed_video(
            video_path,
            timed_clips,
            float(duration or 0),
            job_dir,
            title or "video",
            logs,
        )
        save_project_stage(proj, "assemble", output_path)

        # Generate subtitles SRT file
        srt_path = None
        try:
            srt_gen_path, srt_msg = generate_srt_for_project(proj)
            if srt_gen_path and Path(srt_gen_path).exists():
                srt_path = srt_gen_path
            log(f"📄 {srt_msg}", logs)
        except Exception as e:
            log(f"⚠️  SRT failed: {e}", logs)

        # Copy outputs to clean static filenames in OUTPUT_DIR for download
        final_video_dest = OUTPUT_DIR / "dubbed_video.mp4"
        shutil.copy2(str(output_path), str(final_video_dest))

        final_srt_dest = OUTPUT_DIR / "subtitles.srt"
        if srt_path and Path(srt_path).exists():
            shutil.copy2(str(srt_path), str(final_srt_dest))
        else:
            final_srt_dest = None

        progress(1.0, desc="Done!")
        yield str(output_path), str(final_video_dest), str(final_srt_dest) if final_srt_dest else None, "\n".join(logs)

    except PipelineError as e:
        log(f"❌ [{e.stage}] {e.message}", logs)
        yield None, None, None, "\n".join(logs)
    except Exception as e:
        import traceback
        log(f"❌ Unexpected error: {e}\n{traceback.format_exc()}", logs)
        yield None, None, None, "\n".join(logs)
    finally:
        shutil.rmtree(str(job_dir), ignore_errors=True)
