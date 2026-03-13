"""
LSTM Predictor Benchmark Script
Runs 9 benchmark configurations: 3 corpora × 3 dictionary modes
Saves each run's output to a separate file in benchmark_lstm/ directory.
After all runs, produces a summary table.
"""

import subprocess
import os
import re
import sys
import time

CMIX = "./cmix.exe"
OUTPUT_DIR = "benchmark_lstm"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results_summary.txt")

# 3 corpora
CORPORA = [
    ("input",  "prof_input/input"),
    ("input2", "prof_input/input2"),
    ("enwik7", "prof_input/enwik7"),
]

# 3 dictionary modes: no-preprocess (no dict), compress with english.dic, compress with words_enwik9_optimal.dic
DICT_MODES = [
    ("no_dict",      None,                                 "no-preprocess"),
    ("english_dic",  "dictionary/english.dic",             "compress"),
    ("enwik9_dic",   "dictionary/words_enwik9_optimal.dic","compress"),
]

def parse_output(text):
    """Parse cmix output to extract key metrics."""
    result = {}

    # Match: 51052 bytes -> 17872 bytes in 6.34 s.
    m = re.search(r'(\d+)\s+bytes\s*->\s*(\d+)\s+bytes\s+in\s+([\d.]+)\s+s', text)
    if m:
        result['input_bytes'] = int(m.group(1))
        result['output_bytes'] = int(m.group(2))
        result['time_seconds'] = float(m.group(3))

    # Match: Speed: 64418.93 bits/s (8052.37 bytes/s). 15523.39 ns/bit
    m = re.search(r'Speed:\s+([\d.]+)\s+bits/s\s+\(([\d.]+)\s+bytes/s\)', text)
    if m:
        result['bits_per_sec'] = float(m.group(1))
        result['bytes_per_sec'] = float(m.group(2))

    # Match: Compression ratio: 35.00744%
    m = re.search(r'Compression ratio:\s+([\d.]+)', text)
    if m:
        result['ratio'] = float(m.group(1))

    return result

def run_benchmark(corpus_name, corpus_path, dict_name, dict_path, mode):
    """Run a single benchmark and save output."""
    run_name = f"{corpus_name}_{dict_name}"
    output_file = os.path.join(OUTPUT_DIR, f"{run_name}.txt")
    compressed_file = os.path.join(OUTPUT_DIR, f"{run_name}.bin")

    cmd = [CMIX, "-p", "lstm"]

    if mode == "compress" and dict_path:
        cmd += ["compress", "-d", dict_path, corpus_path, compressed_file]
    else:
        cmd += ["no-preprocess", corpus_path, compressed_file]

    print(f"\n{'='*70}")
    print(f"Running: {run_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=72000)
        elapsed = time.time() - start
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        output = f"TIMEOUT after {elapsed:.1f}s"
    except Exception as e:
        elapsed = time.time() - start
        output = f"ERROR: {e}"

    # Save raw output
    with open(output_file, 'w') as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Wall time: {elapsed:.2f}s\n")
        f.write(f"{'='*70}\n")
        f.write(output)

    print(output[-500:] if len(output) > 500 else output)
    print(f"Wall time: {elapsed:.2f}s")

    return output, elapsed

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Verify cmix exists
    if not os.path.exists(CMIX):
        print(f"Error: {CMIX} not found. Build first with 'make cmix'")
        sys.exit(1)

    all_results = []
    total_start = time.time()

    for corpus_name, corpus_path in CORPORA:
        for dict_name, dict_path, mode in DICT_MODES:
            run_name = f"{corpus_name}_{dict_name}"
            output, wall_time = run_benchmark(
                corpus_name, corpus_path, dict_name, dict_path, mode
            )
            parsed = parse_output(output)
            parsed['run_name'] = run_name
            parsed['corpus'] = corpus_name
            parsed['dict'] = dict_name
            parsed['wall_time'] = wall_time
            all_results.append(parsed)

    total_elapsed = time.time() - total_start

    # Print summary table
    print(f"\n\n{'='*90}")
    print(f"LSTM PREDICTOR BENCHMARK SUMMARY")
    print(f"{'='*90}")
    header = f"{'Run':<30} {'Input':>10} {'Output':>10} {'Ratio':>8} {'Time(s)':>10} {'Speed(B/s)':>12}"
    print(header)
    print("-" * 90)

    summary_lines = [
        "LSTM Predictor Benchmark Results",
        "=" * 90,
        header,
        "-" * 90,
    ]

    for r in all_results:
        inp = r.get('input_bytes', 0)
        out = r.get('output_bytes', 0)
        ratio = r.get('ratio', 0)
        t = r.get('time_seconds', r.get('wall_time', 0))
        spd = r.get('bytes_per_sec', 0)
        line = f"{r['run_name']:<30} {inp:>10} {out:>10} {ratio:>7.3f}% {t:>10.2f} {spd:>12.2f}"
        print(line)
        summary_lines.append(line)

    print(f"\nTotal benchmark time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    summary_lines.append(f"\nTotal benchmark time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Save summary
    with open(RESULTS_FILE, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')

    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"Individual run outputs in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
