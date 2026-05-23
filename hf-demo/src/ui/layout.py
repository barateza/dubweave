import os
import gradio as gr

from src.config import (
    DEMO_MODE,
    DEFAULT_INPUT_LANGUAGE,
    DEFAULT_OUTPUT_LANGUAGE,
    EDGE_TTS_DEFAULT_GENDER,
    EDGE_TTS_VOICE_NAME,
    INPUT_LANGUAGE_OPTIONS,
    OUTPUT_LANGUAGE_CATALOG,
    WHISPER_MODEL,
    get_default_edge_tts_voice,
    get_edge_tts_voice_choices,
)
from src.utils.project import generate_srt_for_project, project_status
from src.utils.system import validate_environment
from src.pipeline import run_pipeline
from src.ui.styles import CSS


def _demo_translation_hint(input_language: str, output_language: str) -> str:
    target_cfg = OUTPUT_LANGUAGE_CATALOG.get(
        output_language, OUTPUT_LANGUAGE_CATALOG[DEFAULT_OUTPUT_LANGUAGE]
    )
    if output_language == "Portuguese (BR)" and input_language in {
        DEFAULT_INPUT_LANGUAGE,
        "English",
    }:
        return (
            "<div style='font-size:0.78rem;opacity:0.85;margin-top:8px;'>"
            "Local fallback (free, CPU-safe) is available for English → Portuguese (BR)."
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


def _build_demo_ui(env_warnings):
    with gr.Blocks(title="Dubweave — HF Spaces Demo", css=CSS) as demo:
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
            label="Upload a video file (Max: 50MB / 3 minutes)",
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
            value="small",
            label="Whisper model",
        )

        openrouter_api_key_input = gr.Textbox(
            label="OpenRouter API key",
            placeholder="Required for language pairs beyond English → Portuguese (BR)",
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

        # States to keep signature alignment with run_pipeline
        url_state = gr.State("")
        speaker_state = gr.State(None)
        browser_state = gr.State("none")
        cookies_state = gr.State(None)
        tts_engine_state = gr.State("Edge TTS (cloud, no key)")
        kokoro_voice_state = gr.State("")
        google_tts_voice_type_state = gr.State("")
        google_tts_voice_name_state = gr.State("")
        supertonic_lang_state = gr.State("")
        supertonic_voice_state = gr.State("")
        gemini_pricing_mode_state = gr.State("auto")
        gemini_single_voice_state = gr.State("")
        gemini_multi_speaker_state = gr.State(False)
        gemini_speaker_assignment_state = gr.State("alternate")
        gemini_speaker1_name_state = gr.State("Speaker1")
        gemini_speaker1_voice_state = gr.State("")
        gemini_speaker2_name_state = gr.State("Speaker2")
        gemini_speaker2_voice_state = gr.State("")
        elevenlabs_voice_id_state = gr.State("")
        elevenlabs_model_id_state = gr.State("")

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
        
        gr.HTML('<div class="panel-label">03 · Outputs</div>')
        video_output = gr.Video(label="Dubbed video player")
        
        with gr.Row():
            video_file_output = gr.File(label="Download Dubbed Video")
            srt_file_output = gr.File(label="Download Subtitles (SRT)")

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
            outputs=[video_output, video_file_output, srt_file_output, log_output],
        )

    return demo


def build_ui():
    env_warnings = validate_environment(demo_mode=DEMO_MODE)
    return _build_demo_ui(env_warnings)
