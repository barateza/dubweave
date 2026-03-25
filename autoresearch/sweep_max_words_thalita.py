import json
import subprocess
from pathlib import Path
import sys

VOICE = "pt-BR-ThalitaNeural"
PROXY_CPS = 21.6 
CONFIG_PATH = Path("autoresearch/merge_config.json")
PIXI_PYTHON = sys.executable

def run_benchmark(description, max_words):
    # Initialize config
    config = {
        "min_words": 10,
        "max_words": max_words,
        "max_duration": None,
        "chars_per_sec": PROXY_CPS,
        "gap_sec": 2.0,
        "voice": VOICE
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    
    cmd = [PIXI_PYTHON, "autoresearch/benchmark.py", "-d", description, "--voice", VOICE]
    print(f"Running: max_words={max_words}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def main():
    print(f"\n=== Sweeping max_words for {VOICE} at CPS={PROXY_CPS} ===")
    print(f"{'MaxW':>6} | {'S':>8} | {'fit':>8} | {'sweet':>8}")
    print("-" * 40)
    
    # Sweep max_words
    for max_w in [50, 70, 100, 150]:
        output = run_benchmark(f"max_words_sweep_thalita: {max_w}", max_w)
        
        s = fit = sweet = "n/a"
        for line in output.split("\n"):
            if "S          =" in line:
                s = line.split("=")[1].split()[0]
            if "fit        =" in line:
                fit = line.split("=")[1].split()[0]
            if "sweet      =" in line:
                sweet = line.split("=")[1].split()[0]
        
        print(f"{max_w:6} | {s:8} | {fit:8} | {sweet:8}")

if __name__ == "__main__":
    main()
