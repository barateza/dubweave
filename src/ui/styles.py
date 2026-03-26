CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:      #0a0a0f;
    --surface: #12121a;
    --card:    #1a1a26;
    --border:  #2a2a3e;
    --accent:  #00e5a0;
    --accent2: #7c6dff;
    --text:    #e8e8f0;
    --muted:   #b0b0cc;
    --danger:  #ff4f6e;
}

* { box-sizing: border-box; }

.skip-link {
    position: absolute;
    top: -40px;
    left: 0;
    background: #0a0a0f;
    color: #00e5a0;
    padding: 8px 16px;
    z-index: 9999;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    border: 1px solid #00e5a0;
    border-radius: 0 0 8px 0;
    text-decoration: none;
}
.skip-link:focus {
    top: 0;
    outline: 2px solid #00e5a0;
    outline-offset: 2px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

.gradio-container { max-width: 900px !important; margin: 0 auto !important; padding: 32px 24px !important; }

#header {
    text-align: center;
    padding: 48px 0 40px;
    position: relative;
}
#header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 40% at 50% 0%, rgba(0,229,160,0.08) 0%, transparent 70%);
    pointer-events: none;
}
#header h1 {
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0;
    background: linear-gradient(135deg, #00e5a0 0%, #7c6dff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#header p {
    color: var(--muted);
    font-size: 1rem;
    margin: 10px 0 0;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

#steps {
    display: flex;
    justify-content: center;
    gap: 0;
    margin-bottom: 36px;
}
.step {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.step-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--border);
    flex-shrink: 0;
}
.step.active .step-dot { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
.step-arrow { color: var(--border); margin: 0 8px; }

.panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,160,0.3), transparent);
}
.panel-label {
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

input[type="text"], textarea, .gr-textbox input, .gr-textbox textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: var(--accent) !important;
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px !important;
    box-shadow: 0 0 0 4px rgba(0,229,160,0.20) !important;
}

label, .gr-form label, .block span {
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.upload-area {
    border: 1.5px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface) !important;
    transition: border-color 0.2s !important;
}
.upload-area:hover { border-color: var(--accent2) !important; }

#run-btn {
    width: 100% !important;
    background: linear-gradient(135deg, #00e5a0, #7c6dff) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    letter-spacing: 0.04em !important;
    padding: 16px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    text-transform: uppercase !important;
}
#run-btn:hover { opacity: 0.92 !important; transform: translateY(-1px) !important; }
#run-btn:active { transform: translateY(0) !important; }
#run-btn:focus-visible { outline: 2px solid #00e5a0 !important; outline-offset: 3px !important; }

#log-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    background: #06060c !important;
    border-color: var(--border) !important;
    color: var(--accent) !important;
    line-height: 1.7 !important;
}

video { border-radius: 12px !important; }

.chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124,109,255,0.1);
    border: 1px solid rgba(124,109,255,0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    color: #a99dff;
    margin: 4px;
}
.chip.green {
    background: rgba(0,229,160,0.08);
    border-color: rgba(0,229,160,0.2);
    color: var(--accent);
}
#chips { margin-bottom: 32px; text-align: center; }

.gr-accordion { background: var(--surface) !important; border-color: var(--border) !important; border-radius: 12px !important; }

@media (forced-colors: active) {
    #header h1 {
        -webkit-text-fill-color: revert;
        color: ButtonText;
        background: none;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}
"""
