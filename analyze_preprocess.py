#!/usr/bin/env python3
"""
Analyze preprocessing approaches for cmix compression on enwik7.

Compares:
  1. Original cmix preprocessing (dictionary.cpp via preprocessor.cpp)
  2. Proposed dictionary_compress.cpp approach

Evaluates: output sizes, byte entropy, alphabet sizes, byte distributions.
"""

import os
import sys
import math
import subprocess
import collections
import struct

INPUT_FILE = "prof_input/enwik7"
DICT_FILE = "dictionary/words_enwik8.dic"

def file_size(path):
    return os.path.getsize(path)

def byte_entropy(data: bytes) -> float:
    """Shannon entropy in bits per byte."""
    if len(data) == 0:
        return 0.0
    freq = collections.Counter(data)
    total = len(data)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def byte_histogram(data: bytes) -> dict:
    return dict(collections.Counter(data))

def alphabet_size(data: bytes) -> int:
    return len(set(data))

def analyze_file(path, label):
    """Print analysis of a preprocessed file."""
    data = open(path, "rb").read()
    size = len(data)
    ent = byte_entropy(data)
    alpha = alphabet_size(data)
    
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")
    print(f"  File:           {path}")
    print(f"  Size:           {size:,} bytes")
    print(f"  Entropy:        {ent:.4f} bits/byte")
    print(f"  Estimated:      {size * ent / 8:,.0f} bytes (entropy lower bound)")
    print(f"  Alphabet size:  {alpha} distinct bytes")
    
    # Show top 20 most frequent bytes
    hist = byte_histogram(data)
    sorted_bytes = sorted(hist.items(), key=lambda x: -x[1])[:20]
    print(f"\n  Top 20 bytes:")
    for b, count in sorted_bytes:
        pct = 100.0 * count / size
        ch = chr(b) if 32 <= b < 127 else f"0x{b:02X}"
        print(f"    byte {b:3d} ({ch:>6s}): {count:8,} ({pct:5.2f}%)")
    
    # Show byte ranges (ASCII text vs control vs high bytes)
    control = sum(v for k, v in hist.items() if k < 32 and k not in (9, 10, 13))
    whitespace = sum(v for k, v in hist.items() if k in (9, 10, 13, 32))
    printable = sum(v for k, v in hist.items() if 33 <= k <= 126)
    high = sum(v for k, v in hist.items() if k >= 128)
    
    print(f"\n  Byte class distribution:")
    print(f"    Control (not ws): {control:8,} ({100.0*control/size:5.2f}%)")
    print(f"    Whitespace:       {whitespace:8,} ({100.0*whitespace/size:5.2f}%)")
    print(f"    Printable ASCII:  {printable:8,} ({100.0*printable/size:5.2f}%)")
    print(f"    High (>=0x80):    {high:8,} ({100.0*high/size:5.2f}%)")
    
    return {"size": size, "entropy": ent, "alpha": alpha}

def check_cmix_exe():
    """Check if cmix.exe exists."""
    if os.path.isfile("cmix.exe"):
        return "cmix.exe"
    return None

def check_dict_compress_exe():
    """Check if dictionary_compress.exe exists."""
    if os.path.isfile("dictionary_compress.exe"):
        return "dictionary_compress.exe"
    return None

def run_original_preprocess(cmix_exe, input_file, dict_file):
    """Run cmix store (preprocessing only) with dictionary."""
    output = "temp_original_preproc.bin"
    cmd = [cmix_exe, "store", input_file, output, "-d", dict_file]
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        print(f"  STDOUT: {result.stdout}")
    return output

def run_original_preprocess_no_dict(cmix_exe, input_file):
    """Run cmix store (preprocessing only) without dictionary."""
    output = "temp_original_preproc_nodict.bin"
    cmd = [cmix_exe, "store", input_file, output]
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        print(f"  STDOUT: {result.stdout}")
    return output

def build_dict_compress():
    """Build dictionary_compress.exe from source."""
    cmd = ["clang++", "-O2", "-std=c++17", "-o", "dictionary_compress.exe", 
           "src/dictionary_compress.cpp"]
    print(f"\nBuilding dictionary_compress: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  Build failed:")
        print(f"  STDERR: {result.stderr}")
        return False
    return True

def run_proposed_preprocess(dc_exe, input_file, dict_path, min_freq=2):
    """Run proposed dictionary_compress build-dict + compress pipeline."""
    # Step 1: Build dictionary from input
    built_dict = "temp_proposed.dict"
    cmd1 = [dc_exe, "build-dict", "--input", input_file, "--dict", built_dict,
            "--suffix", "--min-frequency", str(min_freq)]
    print(f"\nRunning: {' '.join(cmd1)}")
    result = subprocess.run(cmd1, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        print(f"  STDOUT: {result.stdout}")
        return None
    print(f"  {result.stdout.strip()}")
    
    # Step 2: Compress using built dictionary
    output = "temp_proposed_preproc.bin"
    cmd2 = [dc_exe, "compress", "--input", input_file, "--dict", built_dict,
            "--output", output, "--suffix"]
    print(f"\nRunning: {' '.join(cmd2)}")
    result = subprocess.run(cmd2, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr}")
        print(f"  STDOUT: {result.stdout}")
        return None
    
    # Step 3: Verify roundtrip
    roundtrip = "temp_proposed_roundtrip.bin"
    cmd3 = [dc_exe, "decompress", "--input", output, "--dict", built_dict,
            "--output", roundtrip]
    print(f"\nVerifying roundtrip: {' '.join(cmd3)}")
    result = subprocess.run(cmd3, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  Roundtrip decompression failed!")
    else:
        orig_size = file_size(input_file)
        rt_size = file_size(roundtrip)
        if orig_size == rt_size:
            # Compare bytes
            orig_data = open(input_file, "rb").read()
            rt_data = open(roundtrip, "rb").read()
            if orig_data == rt_data:
                print(f"  Roundtrip: VERIFIED OK")
            else:
                # Find first difference
                for i in range(min(len(orig_data), len(rt_data))):
                    if orig_data[i] != rt_data[i]:
                        print(f"  Roundtrip: FAILED at byte {i} (orig={orig_data[i]:02X} rt={rt_data[i]:02X})")
                        ctx_start = max(0, i - 20)
                        ctx_end = min(len(orig_data), i + 20)
                        print(f"    orig context: {orig_data[ctx_start:ctx_end]!r}")
                        print(f"    rt   context: {rt_data[ctx_start:ctx_end]!r}")
                        break
        else:
            print(f"  Roundtrip: SIZE MISMATCH orig={orig_size} rt={rt_size}")
        if os.path.exists(roundtrip):
            os.remove(roundtrip)
    
    return output

def compare_byte_distributions(path1, label1, path2, label2):
    """Compare byte distributions between two preprocessed files."""
    data1 = open(path1, "rb").read()
    data2 = open(path2, "rb").read()
    
    hist1 = collections.Counter(data1)
    hist2 = collections.Counter(data2)
    
    all_bytes = sorted(set(hist1.keys()) | set(hist2.keys()))
    
    print(f"\n{'='*60}")
    print(f" Byte Distribution Comparison")
    print(f"{'='*60}")
    print(f"  {'Byte':>6s} {'Char':>6s} | {label1:>12s} | {label2:>12s} | {'Diff':>12s}")
    print(f"  {'-'*6} {'-'*6} + {'-'*12} + {'-'*12} + {'-'*12}")
    
    # Show only bytes with significant differences
    diffs = []
    for b in all_bytes:
        c1 = hist1.get(b, 0)
        c2 = hist2.get(b, 0)
        diff = c2 - c1
        if abs(diff) > 100:
            ch = chr(b) if 32 <= b < 127 else f"0x{b:02X}"
            diffs.append((abs(diff), b, ch, c1, c2, diff))
    
    diffs.sort(reverse=True)
    for _, b, ch, c1, c2, diff in diffs[:30]:
        sign = "+" if diff > 0 else ""
        print(f"  {b:6d} {ch:>6s} | {c1:12,} | {c2:12,} | {sign}{diff:11,}")

def parse_store_output(path):
    """Parse the cmix store output format to extract the preprocessed content.
    
    Store format: 5 header bytes + 32 vocab bytes (for files >= 10000) + preprocessed data.
    The header contains the original length and a dictionary flag.
    """
    data = open(path, "rb").read()
    if len(data) < 5:
        return data
    
    # Read the 5-byte header
    first_byte = data[0]
    dict_used = bool(first_byte & 0x80)
    first_byte_clean = first_byte & 0x7F
    
    # Reconstruct stored length (but for store mode, length is 0 and actual data follows)
    length = 0
    for i in range(5):
        length <<= 8
        b = data[i]
        if i == 0:
            b &= 0x7F
        length += b
    
    print(f"\n  Store header: dict_used={dict_used}, header_length_field={length}")
    
    if length == 0:
        # Store mode: no compression, just preprocessing
        # Data starts at offset 5
        return data[5:]
    else:
        # Has length field and possibly vocab
        return data  # Return everything for analysis

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if not os.path.isfile(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return 1
    
    input_size = file_size(INPUT_FILE)
    print(f"Input file: {INPUT_FILE} ({input_size:,} bytes)")
    
    # Analyze raw input
    raw_stats = analyze_file(INPUT_FILE, "RAW INPUT (enwik7)")
    
    results = {}
    temp_files = []
    
    # 1. Test original cmix preprocessing
    cmix_exe = check_cmix_exe()
    if cmix_exe:
        print(f"\n{'#'*60}")
        print(f" ORIGINAL CMIX PREPROCESSING")
        print(f"{'#'*60}")
        
        if os.path.isfile(DICT_FILE):
            out_path = run_original_preprocess(cmix_exe, INPUT_FILE, DICT_FILE)
            temp_files.append(out_path)
            if os.path.isfile(out_path):
                # Parse the stored data to get just the preprocessed content
                preproc_data = parse_store_output(out_path)
                # Write just the preprocessed content for analysis
                preproc_path = "temp_orig_preproc_content.bin"
                open(preproc_path, "wb").write(preproc_data)
                temp_files.append(preproc_path)
                results["original_with_dict"] = analyze_file(preproc_path, "ORIGINAL PREPROC (with dictionary)")
                
                # Also analyze the full store output
                results["original_store"] = analyze_file(out_path, "ORIGINAL STORE output (full file)")
        
        # Without dictionary
        out_path_nodict = run_original_preprocess_no_dict(cmix_exe, INPUT_FILE)
        temp_files.append(out_path_nodict)
        if os.path.isfile(out_path_nodict):
            preproc_data_nd = parse_store_output(out_path_nodict)
            preproc_path_nd = "temp_orig_preproc_nodict_content.bin"
            open(preproc_path_nd, "wb").write(preproc_data_nd)
            temp_files.append(preproc_path_nd)
            results["original_no_dict"] = analyze_file(preproc_path_nd, "ORIGINAL PREPROC (no dictionary)")
    else:
        print("\ncmix.exe not found! Build it with 'make cmix' first.")
    
    # 2. Test proposed dictionary_compress preprocessing  
    print(f"\n{'#'*60}")
    print(f" PROPOSED DICTIONARY_COMPRESS PREPROCESSING")
    print(f"{'#'*60}")
    
    dc_exe = check_dict_compress_exe()
    if not dc_exe:
        if not build_dict_compress():
            print("Failed to build dictionary_compress.exe")
            return 1
        dc_exe = "dictionary_compress.exe"
    
    # Test with different min_freq values
    for min_freq in [2, 3, 5]:
        label = f"min_freq={min_freq}"
        out_path = run_proposed_preprocess(dc_exe, INPUT_FILE, "temp_proposed.dict", min_freq=min_freq)
        if out_path and os.path.isfile(out_path):
            temp_files.append(out_path)
            key = f"proposed_{label}"
            results[key] = analyze_file(out_path, f"PROPOSED PREPROC ({label}, suffix)")
    
    # 3. Summary comparison
    print(f"\n{'#'*60}")
    print(f" SUMMARY COMPARISON")
    print(f"{'#'*60}")
    print(f"\n  {'Approach':<45s} {'Size':>12s} {'Entropy':>10s} {'Est.Comp':>12s} {'Ratio':>8s}")
    print(f"  {'-'*45} {'-'*12} {'-'*10} {'-'*12} {'-'*8}")
    
    print(f"  {'Raw input':<45s} {raw_stats['size']:12,} {raw_stats['entropy']:10.4f} {raw_stats['size']*raw_stats['entropy']/8:12,.0f} {'1.000':>8s}")
    
    for key, stats in sorted(results.items()):
        ratio = stats['size'] / raw_stats['size']
        est_comp = stats['size'] * stats['entropy'] / 8
        print(f"  {key:<45s} {stats['size']:12,} {stats['entropy']:10.4f} {est_comp:12,.0f} {ratio:8.3f}")
    
    # 4. Cross-compare byte distributions
    if "original_with_dict" in results and any(k.startswith("proposed_") for k in results):
        proposed_key = [k for k in results if k.startswith("proposed_")][0]
        compare_byte_distributions(
            "temp_orig_preproc_content.bin", "Original",
            "temp_proposed_preproc.bin", "Proposed"
        )
    
    # Cleanup
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    for f in ["temp_proposed.dict", "temp_orig_preproc_content.bin", 
              "temp_orig_preproc_nodict_content.bin"]:
        if os.path.exists(f):
            os.remove(f)
    
    # 5. Key differences analysis
    print(f"\n{'#'*60}")
    print(f" KEY ARCHITECTURAL DIFFERENCES")
    print(f"{'#'*60}")
    print("""
  ORIGINAL cmix preprocessing (preprocessor.cpp + dictionary.cpp):
  ----------------------------------------------------------------
  1. SEGMENT DETECTION: Splits input into TEXT vs DEFAULT segments
     - TEXT: >500 consecutive ASCII chars with spaces => dictionary encoding
     - DEFAULT: Binary/non-text data passed through verbatim
  2. DICTIONARY ENCODING (for TEXT segments):
     - Uses a fixed external dictionary (words_enwik8.dic) 
     - Variable-length byte codes: 1-3 bytes per word
     - Code assignment: sequential by line order in dictionary file
     - Boundary scheme: 80 words @ 1 byte, 3840 @ 2 bytes, 40960 @ 3 bytes
     - Codes use high bit (0x80+) to distinguish from literal ASCII
     - Case handling: single-byte markers (0x40=capitalized, 0x07=uppercase)
     - Substring matching: tries suffix then prefix of long unknown words
  3. CHARACTER REMAPPING: The encoded text gets byte-level remapping to 
     reduce alphabet and improve compression (XOR/swap of certain ranges)
  4. FALLBACK: If dictionary encoding doesn't save >50 bytes, uses raw text
  5. PRETRAINING: Dictionary content is fed to the predictor before compression
     so the model learns word patterns upfront

  PROPOSED dictionary_compress.cpp:
  ---------------------------------
  1. NO SEGMENT DETECTION: Treats entire input as one stream
  2. DICTIONARY: Built from the input itself (not external)
     - base62 codes (a-zA-Z0-9) instead of high-byte codes
     - Codes overlap with the content alphabet!
     - Marker characters: ~ (lower), ^ (cap), * (allcaps), \\ (escape)
  3. CASE: Separate markers for lower/cap/upper per word
  4. SUFFIX SPLITTING: Can split words into root+suffix
  5. ESCAPING: Non-word spans containing markers need length-prefixed escaping
  6. NO PRETRAINING: No dictionary pretraining step
  7. NO CHARACTER REMAPPING: No byte-level remapping to reduce alphabet
""")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
