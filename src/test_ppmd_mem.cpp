/**
 * PPMD Memory Diagnostic Tool
 * 
 * This tool tests PPMD directly at the ppmd_Model level to diagnose
 * memory allocation issues. It processes input files byte-by-byte
 * and reports internal memory statistics.
 * 
 * Usage: test_ppmd_mem <input_file> <order> <memory_mb>
 * 
 * Example: test_ppmd_mem input.txt 25 256
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>

// Include PPMD implementation directly
// This gives us access to ppmd_Model internals
#define PPMD_INTERNAL_TEST
#include "models/ppmd.cpp"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <order> <memory_mb>"
              << std::endl;
    std::cerr << "  input_file: File to process" << std::endl;
    std::cerr << "  order: PPMD order (typically 1-25)" << std::endl;
    std::cerr << "  memory_mb: Memory allocation in MB" << std::endl;
    std::cerr << "\nExample: " << argv[0] << " input.txt 25 256" << std::endl;
    return 1;
  }

  const char *input_file = argv[1];
  int         order      = std::atoi(argv[2]);
  int         memory_mb  = std::atoi(argv[3]);

  if (order < 1 || order > 25) {
    std::cerr << "Error: Order must be between 1 and 25" << std::endl;
    return 1;
  }

  if (memory_mb < 1 || memory_mb > 16384) {
    std::cerr << "Error: Memory must be between 1 and 16384 MB" << std::endl;
    return 1;
  }

  // Open input file
  FILE *input = fopen(input_file, "rb");
  if (!input) {
    std::cerr << "Error: Cannot open input file: " << input_file << std::endl;
    return 1;
  }

  // Get file size
  fseek(input, 0, SEEK_END);
  size_t file_size = ftell(input);
  fseek(input, 0, SEEK_SET);

  std::cout << "\n=== PPMD Memory Diagnostic Tool ===" << std::endl;
  std::cout << "Input file: " << input_file << std::endl;
  std::cout << "File size: " << file_size << " bytes" << std::endl;
  std::cout << "PPMD order: " << order << std::endl;
  std::cout << "Allocated memory: " << memory_mb << " MB (" 
            << (memory_mb * 1024.0 * 1024.0 / 1e9) << " GB)" << std::endl;
  std::cout << std::endl;

  // Create and initialize PPMD model
  PPMD::ppmd_Model model;
  
  std::cout << "Initializing PPMD model..." << std::endl;
  unsigned int init_result = model.Init(order, memory_mb, 1, file_size);
  if (init_result != 0) {
    std::cerr << "Error: PPMD initialization failed!" << std::endl;
    fclose(input);
    return 1;
  }
  
  std::cout << "PPMD model initialized successfully." << std::endl;
  std::cout << std::endl;

  // Performance tracking
  auto start_time = std::chrono::high_resolution_clock::now();
  
  const size_t report_interval = 10240;  // Report every 10KB
  size_t bytes_processed = 0;
  size_t last_report = 0;

  std::cout << "Processing file..." << std::endl;
  std::cout << std::endl;
  
  // Process file byte by byte
  int byte_val;
  while ((byte_val = fgetc(input)) != EOF) {
    unsigned char byte = static_cast<unsigned char>(byte_val);
    
    // Generate predictions for this byte (this exercises the model)
    model.ppmd_PrepareByte();
    
    // Update model with actual byte
    model.ppmd_UpdateByte(byte);
    
    bytes_processed++;
    
    // Report progress
    if (bytes_processed - last_report >= report_interval) {
      double percent = 100.0 * bytes_processed / file_size;
      
      // Get memory statistics
      unsigned long long used_memory = model.GetUsedMemory();
      unsigned long long total_memory = memory_mb * 1024ULL * 1024ULL;
      double memory_usage_percent = 100.0 * used_memory / total_memory;
      
      // Calculate speed
      auto current_time = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          current_time - start_time).count();
      double speed_bps = (bytes_processed * 1000.0) / (elapsed > 0 ? elapsed : 1);
      
      std::cout << "Progress: " << std::fixed << std::setprecision(2) 
                << percent << "% | "
                << "Bytes: " << bytes_processed << " / " << file_size << " | "
                << "Speed: " << std::setprecision(0) << speed_bps << " bytes/s | "
                << "Memory: " << std::setprecision(2) << memory_usage_percent << "% ("
                << (used_memory / 1024.0 / 1024.0) << " / "
                << (total_memory / 1024.0 / 1024.0) << " MB)"
                << std::endl;
      
      last_report = bytes_processed;
    }
  }

  fclose(input);

  // Final statistics
  auto end_time = std::chrono::high_resolution_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time).count();
  double elapsed_sec = elapsed_ms / 1000.0;
  
  unsigned long long final_memory = model.GetUsedMemory();
  unsigned long long total_memory = memory_mb * 1024ULL * 1024ULL;
  
  std::cout << std::endl;
  std::cout << "=== Final Statistics ===" << std::endl;
  std::cout << "Total bytes processed: " << bytes_processed << std::endl;
  std::cout << "Total time: " << std::fixed << std::setprecision(2) 
            << elapsed_sec << " seconds" << std::endl;
  std::cout << "Speed: " << std::setprecision(2) 
            << (bytes_processed / elapsed_sec) << " bytes/s" << std::endl;
  std::cout << "Final memory usage: " << std::setprecision(2)
            << (100.0 * final_memory / total_memory) << "% ("
            << (final_memory / 1024.0 / 1024.0) << " / "
            << (total_memory / 1024.0 / 1024.0) << " MB)" << std::endl;
  std::cout << std::endl;

  return 0;
}
