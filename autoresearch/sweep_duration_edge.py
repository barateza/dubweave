import json
import subprocess
from pathlib import Path
import sys

VOICE = "pt-BR-FranciscaNeural"
PROXY_CPS = 18.3 
CONFIG_PATH = Path("autoresearch/merge_config.json")
PIXI_PYTHON = sys.executable

def run_benchmark(description, max_words, min_words, max_dur):
    # Initialize config
    config = {
        "min_words": min_words,
        "max_words": max_words,
        "max_duration": max_dur,
        "chars_per_sec": PROXY_CPS,
        "gap_sec": 2.0,
        "voice": VOICE
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    
    cmd = [PIXI_PYTHON, "autoresearch/benchmark.py", "-d", description, "--voice", VOICE]
    print(f"Running: {description}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def main():
    print(f"\n=== Sweeping max_duration for {VOICE} at CPS={PROXY_CPS} ===")
    print(f"{'max_dur':>8} | {'S':>8} | {'fit':>8} | {'sweet':>8} | {'boundary':>8}")
    print("-" * 60)
    
    # Sweep max_duration
    for dur in [None, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0]:
        output = run_benchmark(f"max_dur_sweep: {dur}", 100, 10, dur)
        
        s = fit = sweet = boundary = "n/a"
        for line in output.split("\n"):
            if "S          =" in line:
                s = line.split("=")[1].split()[0]
            if "fit        =" in line:
                fit = line.split("=")[1].split()[0]
            if "sweet      =" in line:
                sweet = line.split("=")[1].split()[0]
            if "boundary   =" in line:
                boundary = line.split("=")[1].split()[0]
        
        label = "null" if dur is None else str(dur)
        print(f"{label:>8} | {s:8} | {fit:8} | {sweet:8} | {boundary:8}")

if __name__ == "__main__":
    main()
