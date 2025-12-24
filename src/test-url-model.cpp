#include "models/url-model.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

typedef unsigned char byte;

// Test tool for URL model analysis
// Reads input file byte-by-byte and tracks URL detection statistics
// Usage: test-url-model <input_file>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
    return 1;
  }

  std::ifstream input(argv[1], std::ios::binary);
  if (!input) {
    std::cerr << "Error: Cannot open input file: " << argv[1] << std::endl;
    return 1;
  }

  // Setup vocabulary (all bytes valid for testing)
  std::vector<bool> vocab(256, true);
  unsigned int      bit_context  = 0;

  int               ppmdOrder    = 8;
  int               ppmdMemoryMb = 64;

  // Create URL model
  URLModel url_model(ppmdOrder, ppmdMemoryMb, true, bit_context, vocab);

  std::cout << "Analyzing file: " << argv[1] << std::endl;
  std::cout << "================================================" << std::endl;

  // Process file byte by byte
  unsigned char      byte;
  unsigned long long bytes_processed = 0;

  while (input.read(reinterpret_cast<char *>(&byte), 1)) {
    bit_context = byte;
    url_model.Perceive(0); // Dummy bit processing

    // After each byte update
    if ((bit_context >> 7) == 0) {
      url_model.ByteUpdate();
      bytes_processed++;

      // Progress indicator
      if (bytes_processed % 1000000 == 0) {
        std::cout << "\rProcessed: " << bytes_processed << " bytes"
                  << std::flush;
      }
    }
  }

  std::cout << std::endl;

  // Print statistics
  const auto &stats = url_model.GetStats();

  std::cout << "\n================================================"
            << std::endl;
  std::cout << "URL Model Statistics:" << std::endl;
  std::cout << "================================================" << std::endl;
  std::cout << "Total bytes:        " << stats.total_bytes << std::endl;
  std::cout << "URL bytes:          " << stats.url_bytes << std::endl;
  std::cout << "URLs detected:      " << stats.urls_detected << std::endl;
  std::cout << "False positives:    " << stats.false_positives << std::endl;

  if (stats.total_bytes > 0) {
    double url_percentage = (100.0 * stats.url_bytes) / stats.total_bytes;
    std::cout << "URL percentage:     " << std::fixed << std::setprecision(2)
              << url_percentage << "%" << std::endl;
  }

  if (stats.urls_detected > 0) {
    double avg_url_length =
        (double)stats.url_bytes / (stats.urls_detected - stats.false_positives);
    std::cout << "Avg URL length:     " << std::fixed << std::setprecision(1)
              << avg_url_length << " bytes" << std::endl;

    double false_positive_rate =
        (100.0 * stats.false_positives) / stats.urls_detected;
    std::cout << "False positive rate: " << std::fixed << std::setprecision(1)
              << false_positive_rate << "%" << std::endl;
  }

  std::cout << "================================================" << std::endl;

  input.close();
  return 0;
}
