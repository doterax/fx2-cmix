#!/usr/bin/env python3
"""
Find the best random seed for cmix compression.
Tests seeds in parallel and reports the one that achieves the best compression ratio.
"""

import subprocess
import os
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CompressionResult:
    """Results from a single compression test."""
    seed: int
    compressed_size: int
    compression_time: float
    speed_ns_per_bit: float
    compression_ratio: float
    success: bool
    error: str = ""


def test_seed(seed: int, cmix_path: str, input_file: str, temp_dir: str) -> CompressionResult:
    """
    Test a single seed value.
    
    Args:
        seed: Random seed to test
        cmix_path: Path to cmix executable
        input_file: Input file to compress
        temp_dir: Directory for temporary files
        
    Returns:
        CompressionResult with test results
    """
    output_file = os.path.join(temp_dir, f"temp_seed_{seed}.bin")
    
    try:
        # Run compression with specific seed
        cmd = [cmix_path, "--seed", str(seed), "no-preprocess", input_file, output_file]
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 600 second timeout
        )
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            error_msg = f"Return code: {result.returncode}"
            if result.stderr:
                error_msg += f" | stderr: {result.stderr[:200]}"
            if result.stdout:
                error_msg += f" | stdout: {result.stdout[:200]}"
            return CompressionResult(
                seed=seed,
                compressed_size=0,
                compression_time=elapsed_time,
                speed_ns_per_bit=0.0,
                compression_ratio=0.0,
                success=False,
                error=error_msg
            )
        
        # Parse output for statistics
        output = result.stdout
        compressed_size = 0
        speed_ns_per_bit = 0.0
        compression_ratio = 0.0
        
        # Extract compressed size
        if os.path.exists(output_file):
            compressed_size = os.path.getsize(output_file)
            # Clean up temp file
            try:
                os.remove(output_file)
            except:
                pass
        
        # Check if compression actually worked (look for "num models" in output)
        # If not present, compression failed silently
        if "num models" not in output:
            return CompressionResult(
                seed=seed,
                compressed_size=0,
                compression_time=elapsed_time,
                speed_ns_per_bit=0.0,
                compression_ratio=0.0,
                success=False,
                error="Compression failed (preprocessing error or no models created)"
            )
        
        # Parse speed from output
        for line in output.split('\n'):
            if 'ns/bit' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'ns/bit':
                        try:
                            speed_ns_per_bit = float(parts[i-1])
                        except:
                            pass
            if 'Compression ratio:' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    try:
                        compression_ratio = float(parts[1].strip().rstrip('%'))
                    except:
                        pass
        
        # Additional check: compression ratio should be reasonable (< 100%)
        if compression_ratio >= 100.0 or compression_ratio == 0.0:
            return CompressionResult(
                seed=seed,
                compressed_size=0,
                compression_time=elapsed_time,
                speed_ns_per_bit=0.0,
                compression_ratio=0.0,
                success=False,
                error=f"Invalid compression ratio: {compression_ratio:.2f}%"
            )
        
        return CompressionResult(
            seed=seed,
            compressed_size=compressed_size,
            compression_time=elapsed_time,
            speed_ns_per_bit=speed_ns_per_bit,
            compression_ratio=compression_ratio,
            success=True
        )
        
    except subprocess.TimeoutExpired:
        return CompressionResult(
            seed=seed,
            compressed_size=0,
            compression_time=60.0,
            speed_ns_per_bit=0.0,
            compression_ratio=0.0,
            success=False,
            error="Timeout"
        )
    except Exception as e:
        return CompressionResult(
            seed=seed,
            compressed_size=0,
            compression_time=0.0,
            speed_ns_per_bit=0.0,
            compression_ratio=0.0,
            success=False,
            error=str(e)
        )


def find_best_seed(
    cmix_path: str,
    input_file: str,
    seed_range: Tuple[int, int] = (0, 1024),
    num_workers: int = 10,
    temp_dir: str = "./temp_seed_test"
) -> CompressionResult:
    """
    Find the best random seed for compression.
    
    Args:
        cmix_path: Path to cmix executable
        input_file: Input file to compress
        seed_range: Tuple of (start, end) seeds to test
        num_workers: Number of parallel workers
        temp_dir: Directory for temporary files
        
    Returns:
        Best CompressionResult found
    """
    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Testing seeds from {seed_range[0]} to {seed_range[1]}")
    print(f"Using {num_workers} parallel workers")
    print(f"Input file: {input_file}")
    print("-" * 80)
    
    best_result = None
    results = []
    completed = 0
    total = seed_range[1] - seed_range[0] + 1
    
    # Process seeds in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(test_seed, seed, cmix_path, input_file, temp_dir): seed
            for seed in range(seed_range[0], seed_range[1] + 1)
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if result.success:
                # Update best result (smaller compressed size is better)
                if best_result is None or result.compressed_size < best_result.compressed_size:
                    best_result = result
                    print(f"[{completed}/{total}] New best! Seed {result.seed}: "
                          f"{result.compressed_size} bytes ({result.compression_ratio:.2f}%), "
                          f"{result.speed_ns_per_bit:.2f} ns/bit")
                elif completed % 10 == 0:
                    print(f"[{completed}/{total}] Progress... Current best: seed {best_result.seed} "
                          f"({best_result.compressed_size} bytes)")
            else:
                print(f"[{completed}/{total}] Seed {result.seed} failed: {result.error}")
    
    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except:
        pass
    
    print("-" * 80)
    print("\n=== RESULTS ===")
    print(f"Best seed: {best_result.seed}")
    print(f"Compressed size: {best_result.compressed_size} bytes")
    print(f"Compression ratio: {best_result.compression_ratio:.2f}%")
    print(f"Speed: {best_result.speed_ns_per_bit:.2f} ns/bit")
    print(f"Time: {best_result.compression_time:.2f}s")
    
    # Show top 10 seeds
    successful_results = [r for r in results if r.success]
    successful_results.sort(key=lambda x: x.compressed_size)
    
    print("\n=== TOP 10 SEEDS ===")
    for i, result in enumerate(successful_results[:10], 1):
        print(f"{i:2d}. Seed {result.seed:4d}: {result.compressed_size:6d} bytes "
              f"({result.compression_ratio:6.2f}%), {result.speed_ns_per_bit:8.2f} ns/bit")
    
    return best_result


def main():
    parser = argparse.ArgumentParser(
        description="Find the best random seed for cmix compression"
    )
    parser.add_argument(
        "--cmix",
        default="./cmix.exe" if os.name == 'nt' else "./cmix",
        help="Path to cmix executable (default: ./cmix.exe on Windows, ./cmix on Unix)"
    )
    parser.add_argument(
        "--input",
        default="./prof_input/input",
        help="Input file to test (default: ./prof_input/input)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start seed (default: 0)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=1024,
        help="End seed (default: 1024)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2, use 1 for sequential)"
    )
    parser.add_argument(
        "--temp-dir",
        default="./temp_seed_test",
        help="Temporary directory for test files (default: ./temp_seed_test)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.cmix):
        print(f"Error: cmix executable not found at {args.cmix}")
        return 1
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return 1
    
    # Run seed search
    start_time = time.time()
    best_result = find_best_seed(
        cmix_path=args.cmix,
        input_file=args.input,
        seed_range=(args.start, args.end),
        num_workers=args.workers,
        temp_dir=args.temp_dir
    )
    total_time = time.time() - start_time
    
    print(f"\nTotal search time: {total_time:.2f}s")
    print(f"\nRecommendation: Update runner.cpp default seed to {best_result.seed}")
    
    return 0


if __name__ == "__main__":
    exit(main())
