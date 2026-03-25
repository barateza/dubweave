import json
import subprocess
from pathlib import Path
import sys

VOICE = "pt-BR-FranciscaNeural"
CONFIG_PATH = Path("autoresearch/merge_config.json")
PIXI_PYTHON = sys.executable

def run_benchmark(description, cps):
    # Initialize config with specific cps
    config = {
        "min_words": 8,
        "max_words": 50,
        "max_duration": None,
        "chars_per_sec": cps,
        "gap_sec": 2.0,
        "voice": VOICE
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    
    cmd = [PIXI_PYTHON, "autoresearch/benchmark.py", "-d", description, "--voice", VOICE]
    print(f"Running: CPS={cps}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def main():
    print(f"\n=== Sweeping chars_per_sec for {VOICE} ===")
    print(f"{'CPS':>6} | {'S':>8} | {'fit':>8} | {'sweet':>8}")
    print("-" * 40)
    
    # Sweep from 10 to 30 with 2.0 steps
    for cps in [11.1, 13.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0, 30.0]:
        output = run_benchmark(f"cps_sweep: {cps}", cps)
        
        s = fit = sweet = "n/a"
        for line in output.split("\n"):
            if "S          =" in line:
                s = line.split("=")[1].split()[0]
            if "fit        =" in line:
                fit = line.split("=")[1].split()[0]
            if "sweet      =" in line:
                sweet = line.split("=")[1].split()[0]
        
        print(f"{cps:6.1f} | {s:8} | {fit:8} | {sweet:8}")

if __name__ == "__main__":
    main()
