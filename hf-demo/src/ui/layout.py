import os

import gradio as gr

try:
    import supertonic  # noqa: F401

    SUPERTONIC_AVAILABLE = True
except ImportError:
    SUPERTONIC_AVAILABLE = False

from src.config import (
    DEMO_MODE,
    DEFAULT_INPUT_LANGUAGE,
    DEFAULT_OUTPUT_LANGUAGE,
    EDGE_TTS_DEFAULT_GENDER,
    EDGE_TTS_PT_BR_VOICES,
    EDGE_TTS_VOICE_NAME,
    INPUT_LANGUAGE_OPTIONS,
    ELEVENLABS_API_KEY,
    ELEVENLABS_TTS_MODEL_ID,
    ELEVENLABS_TTS_VOICE_ID,
    GEMINI_TTS_API_KEY,
    GEMINI_TTS_MULTI_SPEAKER,
    GEMINI_TTS_PRICING_MODE,
    GEMINI_TTS_SINGLE_VOICE,
    GEMINI_TTS_SPEAKER1_NAME,
    GEMINI_TTS_SPEAKER1_VOICE,
    GEMINI_TTS_SPEAKER2_NAME,
    GEMINI_TTS_SPEAKER2_VOICE,
    GEMINI_TTS_SPEAKER_ASSIGNMENT,
    GEMINI_TTS_VOICES,
    GOOGLE_TTS_API_KEY,
    GOOGLE_TTS_VOICE_CATALOG,
    GOOGLE_TTS_VOICE_NAME,
    GOOGLE_TTS_VOICE_TYPE,
    KOKORO_VOICE,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    OUTPUT_LANGUAGE_CATALOG,
    WHISPER_MODEL,
    SUPERTONIC_DEFAULT_LANG,
    SUPERTONIC_DEFAULT_VOICE,
    SUPERTONIC_LANGUAGE_OPTIONS,
    SUPERTONIC_VOICE_OPTIONS,
    get_default_edge_tts_voice,
    get_edge_tts_voice_choices,
)
from src.utils.project import generate_srt_for_project, project_status
from src.utils.security import list_elevenlabs_voices
from src.utils.system import validate_environment
from src.core.ingest import get_video_metadata
from src.core.pricing import (
    estimate_google_tts_cost,
    estimate_local_tts_cost,
    estimate_openrouter_translation_cost,
    pick_gemini_tts_cost,
    estimate_elevenlabs_tts_cost,
)
from src.pipeline import run_pipeline
from src.ui.styles import CSS
from src.utils.project import generate_srt_for_project, project_status
from src.utils.system import validate_environment


def _demo_translation_hint(input_language: str, output_language: str) -> str:
    target_cfg = OUTPUT_LANGUAGE_CATALOG.get(
        output_language, OUTPUT_LANGUAGE_CATALOG[DEFAULT_OUTPUT_LANGUAGE]
    )
    if target_cfg.get("supports_local_fallback") and input_language in {
        DEFAULT_INPUT_LANGUAGE,
        "English",
    }:
        return (
            "<div style='font-size:0.78rem;opacity:0.85;margin-top:8px;'>"
            "Local fallback is available for English → Portuguese (BR)."
            "</div>"
        )
    if output_language == "Portuguese (BR)":
        return (
            "<div style='font-size:0.78rem;opacity:0.85;margin-top:8px;'>"
            "Other input languages will need an OpenRouter API key for Portuguese output."
            "</div>"
        )
    return (
        "<div style='font-size:0.78rem;opacity:0.85;margin-top:8px;'>"
        f"{output_language} output requires an OpenRouter API key."
        "</div>"
    )


def _demo_edge_voice_choices(output_language: str, gender: str) -> list[str]:
    choices = get_edge_tts_voice_choices(output_language, gender)
    if choices:
        return choices
    return get_edge_tts_voice_choices(DEFAULT_OUTPUT_LANGUAGE, EDGE_TTS_DEFAULT_GENDER)


def update_cost_info(engine, google_type, meta, gemini_mode, elevenlabs_model):
    parts = []
    dur = meta.get("duration", 0.0)

    if OPENROUTER_API_KEY:
        model_name = OPENROUTER_MODEL.split("/")[-1]
        total_trans = estimate_openrouter_translation_cost(dur, OPENROUTER_MODEL)
        parts.append(
            f"<div><span class='service'>🌐 Translation ({model_name})</span>: ~<strong>${total_trans:.4f}</strong> total est.</div>"
        )

    if engine == "Google Cloud TTS":
        total_tts = estimate_google_tts_cost(dur, google_type)
        parts.append(
            f"<div><span class='service'>🔊 Synthesis ({google_type})</span>: ~<strong>${total_tts:.4f}</strong> total est.</div>"
        )

    if engine == "Gemini 3.1 Flash TTS Preview":
        gemini_est = pick_gemini_tts_cost(dur, gemini_mode)
        parts.append(
            (
                "<div><span class='service'>🔊 Synthesis (Gemini 3.1 Flash TTS Preview)</span>: "
                f"~<strong>${gemini_est.total_cost_usd:.4f}</strong> total est. "
                f"({gemini_est.audio_tokens:.0f} audio tokens, mode: {gemini_est.mode})"
                "</div>"
            )
        )

    if engine == "ElevenLabs TTS":
        total_tts = estimate_elevenlabs_tts_cost(dur)
        model_lbl = (elevenlabs_model or ELEVENLABS_TTS_MODEL_ID).strip()
        parts.append(
            f"<div><span class='service'>🔊 Synthesis (ElevenLabs: {model_lbl})</span>: ~<strong>${total_tts:.4f}</strong> total est.</div>"
        )

    if engine.startswith("Supertonic"):
        local_tts = estimate_local_tts_cost(dur)
        parts.append(
            "<div><span class='service'>🔊 Synthesis (Supertonic v3)</span>: "
            f"~<strong>${local_tts:.4f}</strong> Local (free)</div>"
        )

    if not parts:
        return ""

    header = (
        f"<div style='margin-bottom:8px;font-size:0.7rem;opacity:0.8;'>Estimating for <strong>{dur:.1f}s</strong> ({meta.get('title', 'Unknown')[:30]}...)</div>"
        if dur > 0
        else ""
    )
    return f"<div class='cost-info'>{header}{''.join(parts)}</div>"


def on_input_change(url, upload, engine, g_type, gemini_mode, elevenlabs_model):
    meta = get_video_metadata(url, upload)
    cost = update_cost_info(engine, g_type, meta, gemini_mode, elevenlabs_model)
    return meta, cost


def _build_demo_ui(env_warnings):
    with gr.Blocks(title="Dubweave — HF Spaces Demo") as demo:
        gr.HTML(
            """
            <a href="#main-content" class="skip-link">Skip to main content</a>
            <div id="header"><h1>DUBWEAVE DEMO</h1><p>file upload → language selection → dubbing → subtitles</p></div>
            <div id="chips">
              <span class="chip green">🎙️ Edge TTS</span><span class="chip">🌐 language-aware</span>
              <span class="chip">🎬 video + srt</span><span class="chip">☁️ CPU Spaces</span>
            </div>
            """
        )

        if env_warnings:
            items = "".join(f"<li style='margin-bottom:6px;'>{w}</li>" for w in env_warnings)
            gr.HTML(
                f"<div style=\"background:rgba(255,79,110,0.08);border:1px solid rgba(255,79,110,0.3);border-radius:10px;padding:14px 18px;margin-bottom:16px;font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#ff4f6e;\"><strong>⚠️ Setup Warnings</strong><ul style='margin:8px 0 0;padding-left:20px;line-height:1.8;'>{items}</ul></div>"
            )

        with gr.Row(elem_id="main-content"):
            with gr.Column(scale=3):
                gr.HTML('<div class="panel-label">01 · Project</div>')
                with gr.Row():
                    project_name_input = gr.Textbox(
                        placeholder="my-video", label="Project name", lines=1, scale=2
                    )
                    resume_from_input = gr.Dropdown(
                        choices=["download", "transcribe", "translate", "synthesize", "assemble"],
                        value="download",
                        label="Resume from stage",
                        scale=1,
                    )
                project_status_html = gr.HTML(
                    "<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#9494b2;margin-top:6px;'>Enter a project name to see its status.</div>"
                )

        def refresh_status(name):
            if not name.strip():
                return "<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#9494b2;'>Enter a project name to see its status.</div>"
            status = project_status(name)
            icons = {True: "<span style='color:#00e5a0'>✓</span>", False: "<span style='color:#9494b2'>·</span>"}
            parts = " &nbsp;·&nbsp; ".join(f"{icons[v]} {s}" for s, v in status.items())
            return f"<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#9494b2;margin-top:6px;'>{parts}</div>"

        project_name_input.change(fn=refresh_status, inputs=project_name_input, outputs=project_status_html)

        gr.HTML('<div class="panel-label">02 · Input</div>')
        video_upload_input = gr.File(
            label="Upload a video file",
            file_types=[".mp4", ".mkv", ".webm", ".avi", ".mov"],
            type="filepath",
        )

        with gr.Row():
            input_language_input = gr.Dropdown(
                choices=list(INPUT_LANGUAGE_OPTIONS.keys()),
                value=DEFAULT_INPUT_LANGUAGE,
                label="Input language",
            )
            output_language_input = gr.Dropdown(
                choices=list(OUTPUT_LANGUAGE_CATALOG.keys()),
                value=DEFAULT_OUTPUT_LANGUAGE,
                label="Output language",
            )

        whisper_model_input = gr.Dropdown(
            choices=["small", "base"],
            value=WHISPER_MODEL if WHISPER_MODEL in {"small", "base"} else "small",
            label="Whisper model",
        )

        openrouter_api_key_input = gr.Textbox(
            label="OpenRouter API key",
            placeholder="Optional for language pairs beyond the local fallback",
            type="password",
        )

        with gr.Row():
            edge_gender_input = gr.Radio(
                choices=["Female", "Male"],
                value=EDGE_TTS_DEFAULT_GENDER,
                label="Edge TTS voice gender",
            )
            edge_voice_input = gr.Dropdown(
                choices=_demo_edge_voice_choices(DEFAULT_OUTPUT_LANGUAGE, EDGE_TTS_DEFAULT_GENDER),
                value=get_default_edge_tts_voice(DEFAULT_OUTPUT_LANGUAGE, EDGE_TTS_DEFAULT_GENDER),
                label="Edge TTS voice",
            )

        translation_hint_html = gr.HTML(
            _demo_translation_hint(DEFAULT_INPUT_LANGUAGE, DEFAULT_OUTPUT_LANGUAGE)
        )

        url_state = gr.State("")
        speaker_state = gr.State(None)
        browser_state = gr.State("none")
        cookies_state = gr.State(None)
        tts_engine_state = gr.State("Edge TTS (cloud, no key)")
        kokoro_voice_state = gr.State(KOKORO_VOICE)
        google_tts_voice_type_state = gr.State(GOOGLE_TTS_VOICE_TYPE)
        google_tts_voice_name_state = gr.State(GOOGLE_TTS_VOICE_NAME)
        supertonic_lang_state = gr.State(SUPERTONIC_DEFAULT_LANG)
        supertonic_voice_state = gr.State(SUPERTONIC_DEFAULT_VOICE)
        gemini_pricing_mode_state = gr.State("auto")
        gemini_single_voice_state = gr.State("Kore")
        gemini_multi_speaker_state = gr.State(False)
        gemini_speaker_assignment_state = gr.State("alternate")
        gemini_speaker1_name_state = gr.State("Speaker1")
        gemini_speaker1_voice_state = gr.State("Kore")
        gemini_speaker2_name_state = gr.State("Speaker2")
        gemini_speaker2_voice_state = gr.State("Puck")
        elevenlabs_voice_id_state = gr.State("")
        elevenlabs_model_id_state = gr.State(ELEVENLABS_TTS_MODEL_ID)

        def update_edge_voice(output_language, gender):
            voices = _demo_edge_voice_choices(output_language, gender)
            return gr.update(choices=voices, value=voices[0] if voices else None)

        def update_translation_hint(input_language, output_language):
            return _demo_translation_hint(input_language, output_language)

        def update_language_controls(input_language, output_language, gender):
            return (
                update_edge_voice(output_language, gender),
                update_translation_hint(input_language, output_language),
            )

        output_language_input.change(
            fn=update_language_controls,
            inputs=[input_language_input, output_language_input, edge_gender_input],
            outputs=[edge_voice_input, translation_hint_html],
        )
        edge_gender_input.change(
            fn=lambda output_language, gender: update_edge_voice(output_language, gender),
            inputs=[output_language_input, edge_gender_input],
            outputs=edge_voice_input,
        )
        input_language_input.change(
            fn=lambda input_language, output_language: update_translation_hint(input_language, output_language),
            inputs=[input_language_input, output_language_input],
            outputs=translation_hint_html,
        )

        run_btn = gr.Button("▶  DUB THIS VIDEO", elem_id="run-btn")
        log_output = gr.Textbox(label="Pipeline log", lines=10, interactive=False, elem_id="log-box")
        video_output = gr.Video(label="Dubbed video")

        with gr.Row():
            srt_btn = gr.Button("📝  Generate SRT")
            srt_file_output = gr.File(label="SRT file")
            srt_status = gr.Textbox(label="Status", lines=1, interactive=False)

        srt_btn.click(
            fn=generate_srt_for_project,
            inputs=project_name_input,
            outputs=[srt_file_output, srt_status],
        )

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                url_state,
                video_upload_input,
                speaker_state,
                whisper_model_input,
                browser_state,
                cookies_state,
                project_name_input,
                resume_from_input,
                tts_engine_state,
                kokoro_voice_state,
                google_tts_voice_type_state,
                google_tts_voice_name_state,
                edge_voice_input,
                supertonic_lang_state,
                supertonic_voice_state,
                gemini_pricing_mode_state,
                gemini_single_voice_state,
                gemini_multi_speaker_state,
                gemini_speaker_assignment_state,
                gemini_speaker1_name_state,
                gemini_speaker1_voice_state,
                gemini_speaker2_name_state,
                gemini_speaker2_voice_state,
                elevenlabs_voice_id_state,
                elevenlabs_model_id_state,
                input_language_input,
                output_language_input,
                openrouter_api_key_input,
            ],
            outputs=[video_output, log_output],
        )

    return demo


def build_ui():
    env_warnings = validate_environment(demo_mode=DEMO_MODE)
    elevenlabs_voice_choices = (
        list_elevenlabs_voices(ELEVENLABS_API_KEY) if ELEVENLABS_API_KEY else []
    )
    elevenlabs_default_voice = ELEVENLABS_TTS_VOICE_ID or (
        elevenlabs_voice_choices[0][1] if elevenlabs_voice_choices else ""
    )

    if DEMO_MODE:
        return _build_demo_ui(env_warnings)

    with gr.Blocks(title="Dubweave — PT-BR") as demo:
        gr.HTML("""
        <a href="#main-content" class="skip-link">Skip to main content</a>
        <div id="header"><h1>DUBWEAVE</h1><p>youtube → dubbing → português brasileiro</p></div>
        <div id="chips">
          <span class="chip green">⚡ XTTS v2 · GPU</span><span class="chip green">🎙️ Voice Clone</span>
          <span class="chip">🌐 NLLB-200 · Local</span><span class="chip">🎬 FFmpeg Mux</span>
          <span class="chip">🔊 Whisper Transcription</span>
        </div>
        <div id="steps">
          <span class="step active"><span class="step-dot"></span>Download</span><span class="step-arrow">→</span>
          <span class="step active"><span class="step-dot"></span>Transcribe</span><span class="step-arrow">→</span>
          <span class="step active"><span class="step-dot"></span>Translate</span><span class="step-arrow">→</span>
          <span class="step active"><span class="step-dot"></span>Synthesize</span><span class="step-arrow">→</span>
          <span class="step active"><span class="step-dot"></span>Mux</span>
        </div>
        """)

        if env_warnings:
            items = "".join(
                f"<li style='margin-bottom:6px;'>{w}</li>" for w in env_warnings
            )
            gr.HTML(
                f"<div style=\"background:rgba(255,79,110,0.08);border:1px solid rgba(255,79,110,0.3);border-radius:10px;padding:14px 18px;margin-bottom:16px;font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:#ff4f6e;\"><strong>⚠️ Setup Warnings</strong><ul style='margin:8px 0 0;padding-left:20px;line-height:1.8;'>{items}</ul></div>"
            )

        with gr.Row(elem_id="main-content"):
            with gr.Column(scale=3):
                gr.HTML('<div class="panel-label">01 · Project</div>')
                with gr.Row():
                    project_name_input = gr.Textbox(
                        placeholder="my-video", label="Project name", lines=1, scale=2
                    )
                    resume_from_input = gr.Dropdown(
                        choices=[
                            "download",
                            "transcribe",
                            "translate",
                            "synthesize",
                            "assemble",
                        ],
                        value="download",
                        label="Resume from stage",
                        scale=1,
                    )
                project_status_html = gr.HTML(
                    "<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#9494b2;margin-top:6px;'>Enter a project name to see its status.</div>"
                )

        video_meta_state = gr.State({"title": "Unknown", "duration": 0.0})

        def refresh_status(name):
            if not name.strip():
                return "<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#9494b2;'>Enter a project name to see its status.</div>"
            status = project_status(name)
            icons = {
                True: "<span style='color:#00e5a0'>✓</span>",
                False: "<span style='color:#9494b2'>·</span>",
            }
            parts = " &nbsp;·&nbsp; ".join(f"{icons[v]} {s}" for s, v in status.items())
            return f"<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#9494b2;margin-top:6px;'>{parts}</div>"

        project_name_input.change(
            fn=refresh_status, inputs=project_name_input, outputs=project_status_html
        )

        gr.HTML('<div class="panel-label">02 · Input</div>')
        with gr.Row():
            url_input = gr.Textbox(
                placeholder="https://youtube.com/watch?v=…", label="Video URL", lines=1
            )
        with gr.Row():
            video_upload_input = gr.File(
                label="Upload a video file",
                file_types=[".mp4", ".mkv", ".webm", ".avi", ".mov"],
                type="filepath",
            )

        with gr.Accordion("🎙️ Custom Voice Reference", open=False):
            speaker_input = gr.Audio(
                label="Voice reference clip", type="filepath", sources=["upload"]
            )

        with gr.Accordion("⚙️ Transcription Model", open=False):
            whisper_model_input = gr.Radio(
                choices=["large-v3-turbo", "large-v3"],
                value="large-v3-turbo",
                label="Whisper model",
            )

        with gr.Accordion("🍪 Browser Cookies", open=False):
            with gr.Row():
                browser_input = gr.Radio(
                    choices=["none", "chrome", "firefox", "edge", "brave"],
                    value="none",
                    label="Option A · Browser",
                )
            cookies_file_input = gr.File(
                label="Option B · cookies.txt", file_types=[".txt"], type="filepath"
            )

        with gr.Accordion("🔊 TTS Engine", open=True):
            tts_choices = [
                "Kokoro (fast, PT-BR native)",
                "Edge TTS (cloud, no key)",
                "XTTS v2 (voice clone)",
            ]
            if GOOGLE_TTS_API_KEY:
                tts_choices.append("Google Cloud TTS")
            if SUPERTONIC_AVAILABLE:
                tts_choices.append("Supertonic v3 (local)")
            if GEMINI_TTS_API_KEY:
                tts_choices.append("Gemini 3.1 Flash TTS Preview")
            if ELEVENLABS_API_KEY:
                tts_choices.append("ElevenLabs TTS")

            tts_engine_input = gr.Radio(
                choices=tts_choices,
                value="Kokoro (fast, PT-BR native)",
                label="TTS engine",
            )
            kokoro_voice_input = gr.Dropdown(
                choices=["pf_dora", "pm_alex", "pm_santa"],
                value=KOKORO_VOICE,
                label="Kokoro voice",
                visible=True,
            )
            edge_voice_input = gr.Dropdown(
                choices=EDGE_TTS_PT_BR_VOICES,
                value=EDGE_TTS_VOICE_NAME,
                label="Edge TTS voice",
                visible=False,
            )

            with gr.Column(visible=False) as supertonic_col:
                gr.HTML(
                    "<div style='font-size:0.78rem;opacity:0.8;margin-bottom:8px;'>"
                    "Supertonic auto-downloads model assets on first run unless disabled in .env."
                    "</div>"
                )
                supertonic_lang_input = gr.Dropdown(
                    choices=SUPERTONIC_LANGUAGE_OPTIONS,
                    value=SUPERTONIC_DEFAULT_LANG,
                    label="Supertonic language",
                )
                supertonic_voice_input = gr.Dropdown(
                    choices=SUPERTONIC_VOICE_OPTIONS,
                    value=SUPERTONIC_DEFAULT_VOICE,
                    label="Supertonic voice",
                )

            with gr.Row(visible=False) as google_tts_row:
                google_voice_type_input = gr.Dropdown(
                    choices=list(GOOGLE_TTS_VOICE_CATALOG.keys()),
                    value=GOOGLE_TTS_VOICE_TYPE,
                    label="Google TTS Type",
                    scale=1,
                )
                google_voice_input = gr.Dropdown(
                    choices=GOOGLE_TTS_VOICE_CATALOG.get(
                        GOOGLE_TTS_VOICE_TYPE, [GOOGLE_TTS_VOICE_NAME]
                    ),
                    value=GOOGLE_TTS_VOICE_NAME,
                    label="Google TTS Voice",
                    scale=2,
                )

            with gr.Column(visible=False) as gemini_tts_col:
                gemini_pricing_mode_input = gr.Dropdown(
                    choices=["auto", "standard", "batch"],
                    value=GEMINI_TTS_PRICING_MODE,
                    label="Gemini pricing mode (estimator)",
                )
                gemini_multi_speaker_input = gr.Checkbox(
                    value=GEMINI_TTS_MULTI_SPEAKER,
                    label="Multi-speaker (up to 2)",
                )
                gemini_single_voice_input = gr.Dropdown(
                    choices=GEMINI_TTS_VOICES,
                    value=GEMINI_TTS_SINGLE_VOICE,
                    label="Gemini single-speaker voice",
                    visible=not GEMINI_TTS_MULTI_SPEAKER,
                )
                with gr.Column(visible=GEMINI_TTS_MULTI_SPEAKER) as gemini_multi_row:
                    gemini_speaker_assignment_input = gr.Dropdown(
                        choices=["alternate", "prefix"],
                        value=GEMINI_TTS_SPEAKER_ASSIGNMENT,
                        label="Speaker assignment",
                    )
                    with gr.Row():
                        gemini_speaker1_name_input = gr.Textbox(
                            value=GEMINI_TTS_SPEAKER1_NAME, label="Speaker 1 name"
                        )
                        gemini_speaker1_voice_input = gr.Dropdown(
                            choices=GEMINI_TTS_VOICES,
                            value=GEMINI_TTS_SPEAKER1_VOICE,
                            label="Speaker 1 voice",
                        )
                    with gr.Row():
                        gemini_speaker2_name_input = gr.Textbox(
                            value=GEMINI_TTS_SPEAKER2_NAME, label="Speaker 2 name"
                        )
                        gemini_speaker2_voice_input = gr.Dropdown(
                            choices=GEMINI_TTS_VOICES,
                            value=GEMINI_TTS_SPEAKER2_VOICE,
                            label="Speaker 2 voice",
                        )

            with gr.Column(visible=False) as elevenlabs_tts_col:
                elevenlabs_model_input = gr.Textbox(
                    value=ELEVENLABS_TTS_MODEL_ID,
                    label="ElevenLabs model ID",
                )
                elevenlabs_voice_input = gr.Dropdown(
                    choices=elevenlabs_voice_choices,
                    value=(
                        elevenlabs_default_voice if elevenlabs_default_voice else None
                    ),
                    label="ElevenLabs voice (from your account)",
                    allow_custom_value=False,
                )

            cost_info_html = gr.HTML(
                update_cost_info(
                    "Kokoro (fast, PT-BR native)",
                    GOOGLE_TTS_VOICE_TYPE,
                    {"duration": 0},
                    GEMINI_TTS_PRICING_MODE,
                    ELEVENLABS_TTS_MODEL_ID,
                )
            )

            def on_tts_change(
                engine, g_type, meta, gemini_mode, gemini_multi, elevenlabs_model
            ):
                v_kokoro = gr.update(visible=engine.startswith("Kokoro"))
                v_edge = gr.update(visible=engine.startswith("Edge"))
                v_supertonic = gr.update(visible=engine.startswith("Supertonic"))
                v_google = gr.update(visible=engine.startswith("Google"))
                v_gemini = gr.update(visible=engine.startswith("Gemini"))
                v_elevenlabs = gr.update(visible=engine.startswith("ElevenLabs"))
                v_gemini_single = gr.update(
                    visible=engine.startswith("Gemini") and not gemini_multi
                )
                v_gemini_multi = gr.update(
                    visible=engine.startswith("Gemini") and gemini_multi
                )
                cost = update_cost_info(
                    engine, g_type, meta, gemini_mode, elevenlabs_model
                )
                return (
                    v_kokoro,
                    v_edge,
                    v_supertonic,
                    v_google,
                    v_gemini,
                    v_elevenlabs,
                    v_gemini_single,
                    v_gemini_multi,
                    cost,
                )

            tts_engine_input.change(
                fn=on_tts_change,
                inputs=[
                    tts_engine_input,
                    google_voice_type_input,
                    video_meta_state,
                    gemini_pricing_mode_input,
                    gemini_multi_speaker_input,
                    elevenlabs_model_input,
                ],
                outputs=[
                    kokoro_voice_input,
                    edge_voice_input,
                    supertonic_col,
                    google_tts_row,
                    gemini_tts_col,
                    elevenlabs_tts_col,
                    gemini_single_voice_input,
                    gemini_multi_row,
                    cost_info_html,
                ],
            )
            google_voice_type_input.change(
                fn=lambda e, t, m, gm, em: update_cost_info(e, t, m, gm, em),
                inputs=[
                    tts_engine_input,
                    google_voice_type_input,
                    video_meta_state,
                    gemini_pricing_mode_input,
                    elevenlabs_model_input,
                ],
                outputs=cost_info_html,
            )

            gemini_pricing_mode_input.change(
                fn=lambda e, t, m, gm, em: update_cost_info(e, t, m, gm, em),
                inputs=[
                    tts_engine_input,
                    google_voice_type_input,
                    video_meta_state,
                    gemini_pricing_mode_input,
                    elevenlabs_model_input,
                ],
                outputs=cost_info_html,
            )

            elevenlabs_model_input.change(
                fn=lambda e, t, m, gm, em: update_cost_info(e, t, m, gm, em),
                inputs=[
                    tts_engine_input,
                    google_voice_type_input,
                    video_meta_state,
                    gemini_pricing_mode_input,
                    elevenlabs_model_input,
                ],
                outputs=cost_info_html,
            )

            gemini_multi_speaker_input.change(
                fn=lambda is_multi, engine: (
                    gr.update(visible=engine.startswith("Gemini") and not is_multi),
                    gr.update(visible=engine.startswith("Gemini") and is_multi),
                ),
                inputs=[gemini_multi_speaker_input, tts_engine_input],
                outputs=[gemini_single_voice_input, gemini_multi_row],
            )

            url_input.change(
                fn=on_input_change,
                inputs=[
                    url_input,
                    video_upload_input,
                    tts_engine_input,
                    google_voice_type_input,
                    gemini_pricing_mode_input,
                    elevenlabs_model_input,
                ],
                outputs=[video_meta_state, cost_info_html],
            )
            video_upload_input.change(
                fn=on_input_change,
                inputs=[
                    url_input,
                    video_upload_input,
                    tts_engine_input,
                    google_voice_type_input,
                    gemini_pricing_mode_input,
                    elevenlabs_model_input,
                ],
                outputs=[video_meta_state, cost_info_html],
            )

        run_btn = gr.Button("▶  DUB THIS VIDEO", elem_id="run-btn")
        log_output = gr.Textbox(
            label="Pipeline log", lines=10, interactive=False, elem_id="log-box"
        )
        video_output = gr.Video(label="Dubbed video")

        with gr.Row():
            srt_btn = gr.Button("📝  Generate SRT")
            srt_file_output = gr.File(label="SRT file")
            srt_status = gr.Textbox(label="Status", lines=1, interactive=False)

        srt_btn.click(
            fn=generate_srt_for_project,
            inputs=project_name_input,
            outputs=[srt_file_output, srt_status],
        )

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                url_input,
                video_upload_input,
                speaker_input,
                whisper_model_input,
                browser_input,
                cookies_file_input,
                project_name_input,
                resume_from_input,
                tts_engine_input,
                kokoro_voice_input,
                google_voice_type_input,
                google_voice_input,
                edge_voice_input,
                supertonic_lang_input,
                supertonic_voice_input,
                gemini_pricing_mode_input,
                gemini_single_voice_input,
                gemini_multi_speaker_input,
                gemini_speaker_assignment_input,
                gemini_speaker1_name_input,
                gemini_speaker1_voice_input,
                gemini_speaker2_name_input,
                gemini_speaker2_voice_input,
                elevenlabs_voice_input,
                elevenlabs_model_input,
            ],
            outputs=[video_output, log_output],
        )

    return demo
