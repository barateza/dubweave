import json
import subprocess
from pathlib import Path
import sys

# Using the Loop 1 scoring proxy (25.0) as requested, instead of ground truth.
VOICES = {
    "pt-BR-FranciscaNeural": 25.0,
    "pt-BR-AntonioNeural": 25.0,
    "pt-BR-ThalitaNeural": 25.0,
}

CONFIG_PATH = Path("autoresearch/merge_config.json")

def run_benchmark(voice, description, baseline=False):
    cmd = [sys.executable, "autoresearch/benchmark.py", "-d", description, "--voice", voice]
    if baseline:
        cmd.append("--baseline")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.stdout

def update_config(config):
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

def main():
    # Since all 3 voices use the same proxy now, they should be identical.
    # We run them once each just to have the entries in the log.
    for voice, cps in VOICES.items():
        print(f"\n=== Starting Loop 1 for voice: {voice} (SCORING PROXY cps={cps}) ===")
        
        # 1. Initialize config
        base_config = {
            "min_words": 8,
            "max_words": 50,
            "max_duration": None,
            "chars_per_sec": cps,
            "gap_sec": None,
            "voice": voice
        }
        update_config(base_config)
        
        # 2. Set Baseline (gap_sec=null)
        run_benchmark(voice, f"Baseline for {voice} (proxy cps={cps})", baseline=True)
        
        # 3. Phase 1: gap_sec sweep
        candidates = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
        best_S = -1.0
        best_gap = None
        
        for gap in candidates:
            config = {**base_config, "gap_sec": gap}
            update_config(config)
            output = run_benchmark(voice, f"gap_sec={gap}: phase 1 sweep (proxy)")
            
            # Extract score from output
            for line in output.split("\n"):
                if "S          =" in line:
                    try:
                        current_score = float(line.split("=")[1].split()[0])
                        if current_score > best_S:
                            best_S = current_score
                            best_gap = gap
                    except:
                        pass
                    break

        print(f"Best gap_sec for {voice}: {best_gap} with S={best_S}")

if __name__ == "__main__":
    main()
