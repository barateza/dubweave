import json
import shutil
import time
from pathlib import Path
from typing import Any
from src.config import PROJECTS_DIR, OUTPUT_DIR, JOB_MAX_AGE_H, WORK_DIR

# ── Reading-speed constants ──────────────────────────────────────────────────
SRT_CHARS_PER_SEC = 17.0
SRT_MIN_DURATION = 1.2
SRT_MAX_CHARS = 80
SRT_MERGE_GAP = 0.5
SRT_LINE_WIDTH = 42

def _srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hh, ms = ms // 3_600_000, ms % 3_600_000
    mm, ms = ms // 60_000, ms % 60_000
    ss, ms = ms // 1_000, ms % 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def _wrap_subtitle_line(text: str) -> str:
    if len(text) <= SRT_LINE_WIDTH: return text
    words = text.split()
    mid, pos, best_split, best_dist = len(text) // 2, 0, max(1, len(words) // 2), float("inf")
    for i in range(1, len(words)):
        pos += len(words[i - 1]) + 1
        dist = abs(pos - mid)
        if dist < best_dist: best_dist, best_split = dist, i
    return " ".join(words[:best_split]) + "\n" + " ".join(words[best_split:])

def generate_srt(segments: list, output_path: Path) -> int:
    cues = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text: continue
        start, end = seg["start"], seg["end"]
        if cues:
            prev = cues[-1]
            combined = prev["text"] + " " + text
            if (start - prev["end"]) <= SRT_MERGE_GAP and len(combined) <= SRT_MAX_CHARS:
                prev["text"], prev["end"] = combined, end
                continue
        cues.append({"start": start, "end": end, "text": text})
    for i, cue in enumerate(cues):
        min_dur = max(SRT_MIN_DURATION, len(cue["text"]) / SRT_CHARS_PER_SEC)
        natural_end = cue["start"] + min_dur
        if cue["end"] < natural_end:
            ceiling = cues[i + 1]["start"] - 0.05 if i + 1 < len(cues) else natural_end
            cue["end"] = min(natural_end, ceiling)
    lines = []
    for idx, cue in enumerate(cues, start=1):
        lines.extend([str(idx), f"{_srt_timestamp(cue['start'])} --> {_srt_timestamp(cue['end'])}", _wrap_subtitle_line(cue["text"]), ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return len(cues)

def project_dir(name: str) -> Path:
    safe = "".join(c for c in name.strip() if c.isalnum() or c in " _-")[:60].strip() or "project"
    d = PROJECTS_DIR / safe; d.mkdir(exist_ok=True); return d

def generate_srt_for_project(project_name: str) -> tuple[str | None, str]:
    proj = project_name.strip()
    if not proj: return None, "❌ No project name provided."
    d = project_dir(proj)
    if not (d / "translated.json").exists(): return None, f"❌ No translated.json for project '{proj}'."
    title = proj
    if (d / "meta.json").exists():
        try: title = json.loads((d / "meta.json").read_text(encoding="utf-8")).get("title", proj)
        except Exception: pass
    segments = json.loads((d / "translated.json").read_text(encoding="utf-8"))
    safe_title = "".join(c for c in title if c.isalnum() or c in " _-")[:50]
    out_dir = d / "outputs"; out_dir.mkdir(exist_ok=True)
    srt_path = out_dir / f"{safe_title}_PT-BR.srt"
    n = generate_srt(segments, srt_path)
    shutil.copy2(str(srt_path), str(OUTPUT_DIR / f"{safe_title}_PT-BR.srt"))
    return str(srt_path), f"✅ {n} cues written → {srt_path.name}"

def cleanup_stale_jobs(logs: list) -> list:
    from src.utils.helpers import log
    if not WORK_DIR.exists(): return logs
    now, cleaned = time.time(), 0
    for entry in WORK_DIR.iterdir():
        if entry.is_dir() and (now - entry.stat().st_mtime) > (JOB_MAX_AGE_H * 3600):
            shutil.rmtree(str(entry), ignore_errors=True); cleaned += 1
    if cleaned: log(f"🧹 Cleaned {cleaned} stale job folder(s)", logs)
    return logs

def project_status(name: str) -> dict:
    d = project_dir(name)
    return {
        "download": (d / "video.mp4").exists() and (d / "audio_orig.wav").exists(),
        "transcribe": (d / "segments.json").exists(),
        "translate": (d / "translated.json").exists(),
        "synthesize": (d / "timed_clips.json").exists(),
        "assemble": any((d / "outputs").glob("*.mp4")) if (d / "outputs").exists() else False,
    }

def save_project_stage(name: str, stage: str, data):
    d = project_dir(name)
    if stage == "download":
        v_src, a_src, title, duration = data
        shutil.copy2(str(v_src), str(d / "video.mp4"))
        shutil.copy2(str(a_src), str(d / "audio_orig.wav"))
        (d / "meta.json").write_text(json.dumps({"title": title, "duration": duration}), encoding="utf-8")
    elif stage == "transcribe": (d / "segments.json").write_text(json.dumps(data), encoding="utf-8")
    elif stage == "translate": (d / "translated.json").write_text(json.dumps(data), encoding="utf-8")
    elif stage == "synthesize":
        seg_dst = d / "segments"; seg_dst.mkdir(exist_ok=True); updated = []
        for clip in data:
            src_path, dst_path = Path(clip["path"]), seg_dst / Path(clip["path"]).name
            if src_path.exists() and src_path != dst_path: shutil.copy2(str(src_path), str(dst_path))
            updated.append({**clip, "path": str(dst_path)})
        (d / "timed_clips.json").write_text(json.dumps(updated), encoding="utf-8")
    elif stage == "assemble":
        out_dst = d / "outputs"; out_dst.mkdir(exist_ok=True)
        if Path(data) != (out_dst / Path(data).name): shutil.copy2(data, str(out_dst / Path(data).name))

def load_project_stage(name: str, stage: str) -> Any:
    d = project_dir(name)
    if stage == "download":
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return d / "video.mp4", d / "audio_orig.wav", meta["title"], meta["duration"]
    return json.loads((d / ({"transcribe":"segments.json", "translate":"translated.json", "synthesize":"timed_clips.json"}[stage])).read_text(encoding="utf-8"))
