#include "PPMDPredictor.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

/**
 * Test program for isolated PPMD performance analysis
 *
 * Usage: test_ppmd <input_file> [order] [memory_mb]
 *
 * Example: test_ppmd input.txt 25 1024
 */

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <input_file> [order] [memory_mb]"
              << std::endl;
    std::cerr << "  order: PPMD order (default: 25)" << std::endl;
    std::cerr << "  memory_mb: Memory in MB (default: 1024)" << std::endl;
    return 1;
  }

  const char *input_file = argv[1];
  int         order      = (argc > 2) ? std::atoi(argv[2]) : 25;
  int         memory_mb  = (argc > 3) ? std::atoi(argv[3]) : 1024;

  // Open input file
  std::ifstream input(input_file, std::ios::binary);
  if (!input) {
    std::cerr << "Error: Cannot open input file: " << input_file << std::endl;
    return 1;
  }

  // Get file size
  input.seekg(0, std::ios::end);
  size_t file_size = input.tellg();
  input.seekg(0, std::ios::beg);

  std::cout << "\n=== PPMD Isolated Performance Test ===" << std::endl;
  std::cout << "Input file: " << input_file << std::endl;
  std::cout << "File size: " << file_size << " bytes" << std::endl;
  std::cout << std::endl;

  // Initialize predictor
  PPMDPredictor predictor(order, memory_mb);
  std::cout << std::endl;

  std::cout << "\n=== PPMD initialized ===" << std::endl;
  std::cout << std::endl;

  // Performance tracking
  unsigned long long total_bits     = 0;
  double             total_log_loss = 0.0;
  auto               start_time     = std::chrono::high_resolution_clock::now();

  // Process file bit by bit
  char                     byte;
  unsigned long long       bytes_processed = 0;
  const unsigned long long report_interval = 10240; // Report every N bytes

  while (input.read(&byte, 1)) {
    unsigned char c = static_cast<unsigned char>(byte);

    // Process each bit in the byte (MSB first)
    for (int bit_pos = 7; bit_pos >= 0; --bit_pos) {
      int bit = (c >> bit_pos) & 1;

      // Get prediction
      float p = predictor.Predict();


      // Calculate log loss (cross-entropy)
      float actual_p = bit ? p : (1.0f - p);
      if (actual_p > 0.0f && actual_p < 1.0f) {
        total_log_loss += -std::log2(actual_p);
      }

      // Update model
      predictor.Perceive(bit);

      total_bits++;
    }

    bytes_processed++;

    // Progress report
    if (bytes_processed % report_interval == 0) {
      auto current_time = std::chrono::high_resolution_clock::now();
      auto elapsed      = std::chrono::duration_cast<std::chrono::milliseconds>(
                         current_time - start_time)
                         .count();

      double progress = (100.0 * bytes_processed) / file_size;
      double speed    = (bytes_processed * 1000.0) / elapsed; // bytes/sec
      double avg_bpc  = total_log_loss / bytes_processed;

      std::cout << std::fixed << std::setprecision(2);
      std::cout << "Progress: " << progress << "% | ";
      std::cout << "Speed: " << speed << " bytes/s | ";
      std::cout << "BPC: " << avg_bpc << "      \r" << std::flush;
    }
  }

  input.close();

  // Final statistics
  auto end_time   = std::chrono::high_resolution_clock::now();
  auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

  double elapsed_sec = elapsed_ms / 1000.0;
  double speed_bytes = bytes_processed / elapsed_sec;
  double speed_bits  = total_bits / elapsed_sec;
  double ns_per_bit  = (elapsed_ms * 1e6) / total_bits;
  double avg_bpc     = total_log_loss / bytes_processed;

  std::cout << std::endl << std::endl;
  std::cout << "=== Performance Results ===" << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "Total time: " << elapsed_sec << " seconds" << std::endl;
  std::cout << "Speed: " << speed_bytes << " bytes/s (" << speed_bits
            << " bits/s)" << std::endl;
  std::cout << "Time per bit: " << ns_per_bit << " ns/bit" << std::endl;
  std::cout << std::endl;
  std::cout << "=== Compression Results ===" << std::endl;
  std::cout << "Average bits per character: " << avg_bpc << std::endl;
  std::cout << "Compression ratio: " << (avg_bpc / 8.0 * 100.0) << "%"
            << std::endl;
  std::cout << std::endl;

  // Memory info
  std::cout << "=== Memory Configuration ===" << std::endl;
  std::cout << "PPMD Order: " << predictor.GetOrder() << std::endl;
  std::cout << "PPMD Memory: " << predictor.GetMemoryMB() << " MB" << std::endl;
  std::cout << std::endl;

  return 0;
}
