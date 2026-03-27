import os
import gradio as gr
from src.config import (
    GOOGLE_TTS_API_KEY, GOOGLE_TTS_VOICE_TYPE, GOOGLE_TTS_VOICE_NAME,
    EDGE_TTS_VOICE_NAME, EDGE_TTS_PT_BR_VOICES, KOKORO_VOICE, GOOGLE_TTS_VOICE_CATALOG,
    OPENROUTER_API_KEY, OPENROUTER_MODEL
)
from src.utils.system import validate_environment
from src.utils.project import project_status, generate_srt_for_project
from src.pipeline import run_pipeline
from src.ui.styles import CSS

def update_cost_info(engine, google_type):
    parts = []
    
    # OpenRouter (Translation)
    if OPENROUTER_API_KEY:
        model_name = OPENROUTER_MODEL.split("/")[-1]
        # Unit prices for Gemini 2.0 Flash on OpenRouter
        cost = "<strong>$0.10</strong>/1M in · <strong>$0.40</strong>/1M out"
        if "flash-lite" in OPENROUTER_MODEL.lower():
            cost = "<strong>$0.075</strong>/1M in · <strong>$0.30</strong>/1M out"
        parts.append(f"<div><span class='service'>🌐 Translation ({model_name})</span>: {cost} (tokens)</div>")

    # Google TTS (Synthesis)
    if engine == "Google Cloud TTS":
        costs = {
            "Standard": "$4.00",
            "WaveNet": "$16.00",
            "Neural2": "$16.00",
            "Studio": "$160.00",
            "Chirp3 HD": "$30.00",
            "Polyglot (Preview)": "$16.00"
        }
        rate = costs.get(google_type, "$16.00")
        parts.append(f"<div><span class='service'>🔊 Synthesis ({google_type})</span>: <strong>{rate}</strong>/1M characters</div>")
    
    if not parts:
        return ""
    
    return f"<div class='cost-info'>{''.join(parts)}</div>"

def build_ui():
    env_warnings = validate_environment()

    with gr.Blocks(title="Dubweave — PT-BR", css=CSS) as demo:
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
            items = "".join(f"<li style='margin-bottom:6px;'>{w}</li>" for w in env_warnings)
            gr.HTML(f'<div style="background:rgba(255,79,110,0.08);border:1px solid rgba(255,79,110,0.3);border-radius:10px;padding:14px 18px;margin-bottom:16px;font-family:\'JetBrains Mono\',monospace;font-size:0.8rem;color:#ff4f6e;"><strong>⚠️ Setup Warnings</strong><ul style=\'margin:8px 0 0;padding-left:20px;line-height:1.8;\'>{items}</ul></div>')

        with gr.Row(elem_id="main-content"):
            with gr.Column(scale=3):
                gr.HTML('<div class="panel-label">01 · Project</div>')
                with gr.Row():
                    project_name_input = gr.Textbox(placeholder="my-video", label="Project name", lines=1, scale=2)
                    resume_from_input = gr.Dropdown(choices=["download", "transcribe", "translate", "synthesize", "assemble"], value="download", label="Resume from stage", scale=1)
                project_status_html = gr.HTML("<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#9494b2;margin-top:6px;'>Enter a project name to see its status.</div>")

        def refresh_status(name):
            if not name.strip(): return "<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#9494b2;'>Enter a project name to see its status.</div>"
            status = project_status(name); icons = {True: "<span style='color:#00e5a0'>✓</span>", False: "<span style='color:#9494b2'>·</span>"}
            parts = " &nbsp;·&nbsp; ".join(f"{icons[v]} {s}" for s, v in status.items())
            return f"<div style='font-size:0.75rem;font-family:JetBrains Mono,monospace;color:#9494b2;margin-top:6px;'>{parts}</div>"

        project_name_input.change(fn=refresh_status, inputs=project_name_input, outputs=project_status_html)

        gr.HTML('<div class="panel-label">02 · Input</div>')
        with gr.Row():
            url_input = gr.Textbox(placeholder="https://youtube.com/watch?v=…", label="Video URL", lines=1)
        with gr.Row():
            video_upload_input = gr.File(label="Upload a video file", file_types=[".mp4", ".mkv", ".webm", ".avi", ".mov"], type="filepath")

        with gr.Accordion("🎙️ Custom Voice Reference", open=False):
            speaker_input = gr.Audio(label="Voice reference clip", type="filepath", sources=["upload"])

        with gr.Accordion("⚙️ Transcription Model", open=False):
            whisper_model_input = gr.Radio(choices=["large-v3-turbo", "large-v3"], value="large-v3-turbo", label="Whisper model")

        with gr.Accordion("🍪 Browser Cookies", open=False):
            with gr.Row():
                browser_input = gr.Radio(choices=["none", "chrome", "firefox", "edge", "brave"], value="none", label="Option A · Browser")
            cookies_file_input = gr.File(label="Option B · cookies.txt", file_types=[".txt"], type="filepath")

        with gr.Accordion("🔊 TTS Engine", open=True):
            tts_engine_input = gr.Radio(choices=["Kokoro (fast, PT-BR native)", "Edge TTS (cloud, no key)", "XTTS v2 (voice clone)"] + (["Google Cloud TTS"] if GOOGLE_TTS_API_KEY else []), value="Kokoro (fast, PT-BR native)", label="TTS engine")
            kokoro_voice_input = gr.Dropdown(choices=["pf_dora", "pm_alex", "pm_santa"], value=KOKORO_VOICE, label="Kokoro voice", visible=True)
            edge_voice_input = gr.Dropdown(choices=EDGE_TTS_PT_BR_VOICES, value=EDGE_TTS_VOICE_NAME, label="Edge TTS voice", visible=False)
            
            with gr.Row(visible=False) as google_tts_row:
                google_voice_type_input = gr.Dropdown(choices=list(GOOGLE_TTS_VOICE_CATALOG.keys()), value=GOOGLE_TTS_VOICE_TYPE, label="Google TTS Type", scale=1)
                google_voice_input = gr.Dropdown(choices=GOOGLE_TTS_VOICE_CATALOG.get(GOOGLE_TTS_VOICE_TYPE, [GOOGLE_TTS_VOICE_NAME]), value=GOOGLE_TTS_VOICE_NAME, label="Google TTS Voice", scale=2)

            cost_info_html = gr.HTML(update_cost_info("Kokoro (fast, PT-BR native)", GOOGLE_TTS_VOICE_TYPE))

            def on_tts_change(engine, g_type):
                v_kokoro = gr.update(visible=engine.startswith("Kokoro"))
                v_edge = gr.update(visible=engine.startswith("Edge"))
                v_google = gr.update(visible=engine.startswith("Google"))
                cost = update_cost_info(engine, g_type)
                return v_kokoro, v_edge, v_google, cost
            
            tts_engine_input.change(fn=on_tts_change, inputs=[tts_engine_input, google_voice_type_input], outputs=[kokoro_voice_input, edge_voice_input, google_tts_row, cost_info_html])
            google_voice_type_input.change(fn=lambda e, t: update_cost_info(e, t), inputs=[tts_engine_input, google_voice_type_input], outputs=cost_info_html)

        run_btn = gr.Button("▶  DUB THIS VIDEO", elem_id="run-btn")
        log_output = gr.Textbox(label="Pipeline log", lines=10, interactive=False, elem_id="log-box")
        video_output = gr.Video(label="Dubbed video")

        with gr.Row():
            srt_btn = gr.Button("📝  Generate SRT")
            srt_file_output = gr.File(label="SRT file")
            srt_status = gr.Textbox(label="Status", lines=1, interactive=False)
        
        srt_btn.click(fn=generate_srt_for_project, inputs=project_name_input, outputs=[srt_file_output, srt_status])

        run_btn.click(fn=run_pipeline, inputs=[url_input, video_upload_input, speaker_input, whisper_model_input, browser_input, cookies_file_input, project_name_input, resume_from_input, tts_engine_input, kokoro_voice_input, google_voice_type_input, google_voice_input, edge_voice_input], outputs=[video_output, log_output])

    return demo
