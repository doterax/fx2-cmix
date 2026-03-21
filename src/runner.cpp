#include <cstdlib>
#include <ctime>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <vector>

#include "coder/decoder.h"
#include "coder/encoder.h"
#include "predictor.h"
#include "generic_full_predictor.h"
#include "preprocess/preprocessor.h"

#include "readalike_prepr/article_reorder.h"
#include "readalike_prepr/misc.h"
#include "readalike_prepr/phda9_preprocess.h"
#include "readalike_prepr/self_extract.h"

#include "random.hpp"
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <chrono>

#include "CLI/CLI11.hpp"

#include "PPMDPredictor.hpp"
#include "LstmPredictor.hpp"
#include "ByteMixerPredictor.hpp"
#include "MixPredictor.hpp"

enum class EPredictorType { FULL, PPMD_ONLY, GENERIC, LSTM_ONLY, BYTE_MIXER, MIX };

namespace {
const int kMinVocabFileSize = 10000;
}

size_t getFileSize(const std::string &path) {
  // // get the size of the output file
  FILE *f = nullptr;
  if (fopen_s(&f, path.c_str(), "rb") != 0 || !f) {
    printf("can't open file for measuring its size");
    return 0;
  }
  fseek(f, 0, SEEK_END);
  size_t output_size = ftell(f);
  fclose(f);
  return output_size;
}

void WriteHeader(unsigned long long length, const std::vector<bool> &vocab,
                 bool dictionary_used, std::ofstream *os) {
  for (int i = 4; i >= 0; --i) {
    char c = length >> (8 * i);
    if (i == 4) {
      c &= 0x7F;
      if (dictionary_used)
        c |= 0x80;
    }
    os->put(c);
  }
  if (length < kMinVocabFileSize)
    return;
  for (int i = 0; i < 32; ++i) {
    unsigned char c = 0;
    for (int j = 0; j < 8; ++j) {
      if (vocab[i * 8 + j])
        c += 1 << j;
    }
    os->put(c);
  }
}

void WriteStorageHeader(FILE *out, bool dictionary_used) {
  for (int i = 4; i >= 0; --i) {
    char c = 0;
    if (i == 4 && dictionary_used)
      c = 0x80;
    putc(c, out);
  }
}

void ReadHeader(std::ifstream *is, unsigned long long *length,
                bool *dictionary_used, std::vector<bool> *vocab) {
  *length = 0;
  for (int i = 0; i <= 4; ++i) {
    *length <<= 8;
    unsigned char c = is->get();
    if (i == 0) {
      if (c & 0x80)
        *dictionary_used = true;
      else
        *dictionary_used = false;
      c &= 0x7F;
    }
    *length += c;
  }
  if (*length == 0)
    return;
  if (*length < kMinVocabFileSize) {
    std::fill(vocab->begin(), vocab->end(), true);
    return;
  }
  for (int i = 0; i < 32; ++i) {
    unsigned char c = is->get();
    for (int j = 0; j < 8; ++j) {
      if (c & (1 << j))
        (*vocab)[i * 8 + j] = true;
    }
  }
}

std::unique_ptr<IPredictor> CreateFullPredictor(const std::vector<bool> &vocab,
                                                int ppmd_order, int ppmd_mb,
                                                int lstm_bptt_depth,
                                                int lstm_bptt_period,
                                                int lstm_cells,
                                                int lstm_layers) {
  return std::make_unique<Predictor>(vocab, ppmd_order, ppmd_mb, lstm_bptt_depth, lstm_bptt_period, lstm_cells, lstm_layers);
}

std::unique_ptr<IPredictor> CreatePPMdPredictor(const std::vector<bool> &vocab,
                                                int ppmd_order, int ppmd_mb) {
  return std::make_unique<PPMDPredictor>(ppmd_order, ppmd_mb);
}

// Generic predictor rebuilt from scratch (no use of Predictor class).
std::unique_ptr<IPredictor> CreateGenericPredictor(const std::vector<bool> &vocab,
                                                   int ppmd_order, int ppmd_mb,
                                                   bool mw_enable,
                                                   float mw_alpha,
                                                   float mw_min,
                                                   float mw_max,
                                                   int lstm_bptt_depth,
                                                   int lstm_bptt_period) {
  auto p = std::make_unique<GenericFullPredictor>(vocab, ppmd_order, ppmd_mb, lstm_bptt_depth, lstm_bptt_period);
  // Explicitly wire all components to reflect flexible initialization
  p->InitFxcm();            // FXCM model
  p->AddBracket();          // Bracket + related contexts
  p->AddPPMD();             // Byte model (PPMD)
  // p->AddURLModel();         // URL-specific predictor
  p->AddWord();             // Word-related models
  p->AddMatch();            // Match models
  p->AddDoubleIndirect();   // Double indirect models
  p->SetAuxiliarySize(2);   // fxcm + byte_mixer
  p->AddMixers();           // Mixers, layers, SSE
  if (mw_enable) {
    p->EnableMixerWeighting(mw_alpha, mw_min, mw_max);
  }
  return p;
}

std::unique_ptr<IPredictor> CreatePredictor(const std::vector<bool> &vocab,
                                            EPredictorType type, int ppmd_order,
                                            int ppmd_mb,
                                            bool mw_enable,
                                            float mw_alpha,
                                            float mw_min,
                                            float mw_max,
                                            int lstm_bptt_depth,
                                            int lstm_bptt_period,
                                            int lstm_cells,
                                            int lstm_layers) {
  if (type == EPredictorType::FULL) {
    return CreateFullPredictor(vocab, ppmd_order, ppmd_mb, lstm_bptt_depth, lstm_bptt_period, lstm_cells, lstm_layers);
  } else if (type == EPredictorType::PPMD_ONLY) {
    return CreatePPMdPredictor(vocab, ppmd_order, ppmd_mb);
  } else if (type == EPredictorType::GENERIC) {
    return CreateGenericPredictor(vocab, ppmd_order, ppmd_mb,
                                  mw_enable, mw_alpha, mw_min, mw_max,
                                  lstm_bptt_depth, lstm_bptt_period);
  } else if (type == EPredictorType::LSTM_ONLY) {
    return std::make_unique<LstmPredictor>(lstm_cells, lstm_layers,
                                           128, 0.03f, 10.0f,
                                           lstm_bptt_depth, lstm_bptt_period,
                                           vocab);
  } else if (type == EPredictorType::BYTE_MIXER) {
    return std::make_unique<ByteMixerPredictor>(ppmd_order, ppmd_mb,
                                                lstm_cells, lstm_layers,
                                                128, 0.03f, 10.0f,
                                                lstm_bptt_depth, lstm_bptt_period,
                                                vocab);
  } else if (type == EPredictorType::MIX) {
    return std::make_unique<MixPredictor>(ppmd_order, ppmd_mb,
                                         lstm_cells, lstm_layers,
                                         128, 0.03f, 10.0f,
                                         lstm_bptt_depth, lstm_bptt_period,
                                         vocab);
  }
  throw std::invalid_argument("Unknown predictor type");
}

void ExtractVocab(unsigned long long num_bytes, std::ifstream *is,
                  std::vector<bool> *vocab) {
  for (size_t pos = 0; pos < num_bytes; ++pos) {
    unsigned char c = is->get();
    (*vocab)[c]     = true;
  }
  assert(num_bytes >= 2);
  Eigen::VectorXi byte_map = Eigen::VectorXi::Zero(256);
  uint16_t        offset   = 0;
  for (int i = 0; i < 256; ++i) {
    byte_map[i] = offset;
    if ((*vocab)[i])
      ++offset;
  }
}

void ClearOutput() {
  fprintf(stderr, "\r                     \r");
  fflush(stderr);
}

void Compress(unsigned long long input_bytes, std::ifstream *is,
              std::ofstream *os, unsigned long long *output_bytes,
              IPredictor *p) {
  Encoder            e(os, p);

  FILE              *progress = nullptr;
  if (fopen_s(&progress, "./progress.log", "w") != 0 || !progress) {
    progress = nullptr; // Disable progress file on failure
  }
  unsigned long long percent  = std::max(512ull, 1 + (input_bytes / 10000));
  ClearOutput();
  auto start_tp = std::chrono::steady_clock::now();
  size_t buffer_size     = 1 * 256 * 1024;
  size_t bytes_remaining = (size_t)input_bytes;
  char  *buffer          = new char[buffer_size];
  is->read(buffer, std::min(bytes_remaining, buffer_size));
  bytes_remaining -= std::min(bytes_remaining, buffer_size);
  for (unsigned long long pos = 0; pos < input_bytes; ++pos) {
    unsigned char c = buffer[pos % buffer_size];
    for (int j = 7; j >= 0; --j) {
      e.Encode((c >> j) & 1);
    }
    if (pos % buffer_size == buffer_size - 1) {
      is->read(buffer, std::min(bytes_remaining, buffer_size));
      bytes_remaining -= std::min(bytes_remaining, buffer_size);
    }
    if (pos % percent == 0) {
      double frac = 100.0 * pos / input_bytes;
      // Compute instantaneous rate and ETA
      auto   now_tp          = std::chrono::steady_clock::now();
      double elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(now_tp - start_tp).count();
      unsigned long long processed_bytes = pos + 1; // bytes processed so far
      double rate_bytes_per_sec          = (elapsed_seconds > 0.0) ? (processed_bytes / elapsed_seconds) : 0.0;
      double eta_seconds                 = (rate_bytes_per_sec > 0.0)
                                               ? (double)(input_bytes - processed_bytes) / rate_bytes_per_sec
                                               : 0.0;
      unsigned long long eta_total_secs  = (unsigned long long)(eta_seconds + 0.5);
      unsigned long long eta_h           = eta_total_secs / 3600ull;
      unsigned long long eta_m           = (eta_total_secs % 3600ull) / 60ull;
      unsigned long long eta_s           = eta_total_secs % 60ull;
      char                eta_buf[32];
      if (eta_h == 0ull) {
        snprintf(eta_buf, sizeof(eta_buf), "%02llu:%02llu", eta_m, eta_s);
      } else {
        snprintf(eta_buf, sizeof(eta_buf), "%llu:%02llu:%02llu", eta_h, eta_m, eta_s);
      }
      fprintf(stderr, "\rprogress: %.2f%% | rate: %1.2f bytes/s | ETA: %s     ", frac, rate_bytes_per_sec, eta_buf);
      fflush(stderr);

      if (progress) {
        fprintf(progress, "%.2f %zu\n", frac, e.OutputSize());
        fflush(progress);
      }
    }
  }
  e.Flush();
  *output_bytes = os->tellp();
  delete[] buffer;
  // Finish the progress line cleanly to avoid overlap with final summary
  fprintf(stderr, "\n");
  fflush(stderr);
  if (progress) {
    fclose(progress);
  }
}

void Decompress(unsigned long long output_length, std::ifstream *is,
                std::ofstream *os, IPredictor *p) {
  Decoder            d(is, p);
  unsigned long long percent = 1 + (output_length / 10000);
  ClearOutput();
  for (unsigned long long pos = 0; pos < output_length; ++pos) {
    int byte = 1;
    while (byte < 256) {
      byte += byte + d.Decode();
    }
    os->put(byte);
    if (pos % percent == 0) {
      double frac = 100.0 * pos / output_length;
      fprintf(stderr, "\rprogress: %.2f%%", frac);
      fflush(stderr);
    }
  }
  // Finish the progress line cleanly to avoid overlap with final summary
  fprintf(stderr, "\n");
  fflush(stderr);
}

bool Store(const std::string &input_path, const std::string &temp_path,
           const std::string &output_path, FILE *dictionary,
           unsigned long long *input_bytes, unsigned long long *output_bytes) {
  FILE *data_in = nullptr;
  if (fopen_s(&data_in, input_path.c_str(), "rb") != 0 || !data_in)
    return false;
  FILE *data_out = nullptr;
  if (fopen_s(&data_out, output_path.c_str(), "wb") != 0 || !data_out)
    return false;
  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);
  WriteStorageHeader(data_out, dictionary != NULL);
  fprintf(stderr, "\rpreprocessing...");
  fflush(stderr);
  preprocessor::Encode(data_in, data_out, *input_bytes, temp_path, dictionary);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(data_in);
  fclose(data_out);
  return true;
}

bool RunCompression(EPredictorType predictor_type, bool enable_preprocess,
                    const std::string &input_path, const std::string &temp_path,
                    const std::string &output_path, FILE *dictionary,
                    unsigned long long *input_bytes,
                    unsigned long long *output_bytes, int ppmd_order,
                    int ppmd_mb,
                    bool mw_enable,
                    double mw_alpha,
                    double mw_min,
                    double mw_max,
                    int lstm_bptt_depth,
                    int lstm_bptt_period,
                    int lstm_cells,
                    int lstm_layers) {
  FILE *data_in = nullptr;
  if (fopen_s(&data_in, input_path.c_str(), "rb") != 0 || !data_in)
    return false;
  FILE *temp_out = nullptr;
  if (fopen_s(&temp_out, temp_path.c_str(), "wb") != 0 || !temp_out)
    return false;

  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);

  if (enable_preprocess) {
    fprintf(stderr, "\rpreprocessing...");
    fflush(stderr);
    preprocessor::Encode(data_in, temp_out, *input_bytes, temp_path,
                         dictionary);
  } else {
    preprocessor::NoPreprocess(data_in, temp_out, *input_bytes);
  }
  fclose(data_in);
  fclose(temp_out);

  std::ifstream temp_in(temp_path, std::ios::in | std::ios::binary);
  if (!temp_in.is_open())
    return false;

  std::ofstream data_out(output_path, std::ios::out | std::ios::binary);
  if (!data_out.is_open())
    return false;

  temp_in.seekg(0, std::ios::end);
  unsigned long long temp_bytes = temp_in.tellg();
  temp_in.seekg(0, std::ios::beg);

  std::vector<bool> vocab(256, false);
  if (temp_bytes < kMinVocabFileSize) {
    std::fill(vocab.begin(), vocab.end(), true);
  } else {
    ExtractVocab(temp_bytes, &temp_in, &vocab);
    temp_in.seekg(0, std::ios::beg);
  }

  WriteHeader(temp_bytes, vocab, dictionary != NULL, &data_out);
  auto p = CreatePredictor(vocab, predictor_type, ppmd_order, ppmd_mb,
                           mw_enable, (float)mw_alpha, (float)mw_min, (float)mw_max,
                           lstm_bptt_depth, lstm_bptt_period,
                           lstm_cells, lstm_layers);
  if (enable_preprocess)
    preprocessor::Pretrain(p.get(), dictionary);
  Compress(temp_bytes, &temp_in, &data_out, output_bytes, p.get());
  temp_in.close();
  data_out.close();
  remove(temp_path.c_str());
  return true;
}

bool RunDecompression(EPredictorType     predictor_type,
                      const std::string &input_path,
                      const std::string &temp_path,
                      const std::string &output_path, FILE *dictionary,
                      unsigned long long *input_bytes,
                      unsigned long long *output_bytes, int ppmd_order,
                      int ppmd_mb,
                      bool mw_enable,
                      double mw_alpha,
                      double mw_min,
                      double mw_max,
                      int lstm_bptt_depth,
                      int lstm_bptt_period,
                      int lstm_cells,
                      int lstm_layers) {
  std::ifstream data_in(input_path, std::ios::in | std::ios::binary);
  if (!data_in.is_open())
    return false;

  data_in.seekg(0, std::ios::end);
  *input_bytes = data_in.tellg();
  data_in.seekg(0, std::ios::beg);
  std::vector<bool> vocab(256, false);
  bool              dictionary_used;
  ReadHeader(&data_in, output_bytes, &dictionary_used, &vocab);
  if (!dictionary_used && dictionary != NULL)
    return false;
  if (dictionary_used && dictionary == NULL)
    return false;

  if (*output_bytes == 0) { // undo store
    data_in.close();
    FILE *in = nullptr;
    if (fopen_s(&in, input_path.c_str(), "rb") != 0 || !in)
      return false;
    FILE *data_out = nullptr;
    if (fopen_s(&data_out, output_path.c_str(), "wb") != 0 || !data_out)
      return false;
    fseek(in, 5L, SEEK_SET);
    fprintf(stderr, "\rdecoding...");
    fflush(stderr);
    preprocessor::Decode(in, data_out, dictionary);
    fseek(data_out, 0L, SEEK_END);
    *output_bytes = ftell(data_out);
    fclose(in);
    fclose(data_out);
    return true;
  }
  auto p = CreatePredictor(vocab, predictor_type, ppmd_order, ppmd_mb,
                           mw_enable, (float)mw_alpha, (float)mw_min, (float)mw_max,
                           lstm_bptt_depth, lstm_bptt_period,
                           lstm_cells, lstm_layers);
  if (dictionary_used)
    preprocessor::Pretrain(p.get(), dictionary);

  std::ofstream temp_out(temp_path, std::ios::out | std::ios::binary);
  if (!temp_out.is_open())
    return false;

  Decompress(*output_bytes, &data_in, &temp_out, p.get());
  data_in.close();
  temp_out.close();

  FILE *temp_in = nullptr;
  if (fopen_s(&temp_in, temp_path.c_str(), "rb") != 0 || !temp_in)
    return false;
  FILE *data_out = nullptr;
  if (fopen_s(&data_out, output_path.c_str(), "wb") != 0 || !data_out)
    return false;

  preprocessor::Decode(temp_in, data_out, dictionary);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(temp_in);
  fclose(data_out);
  remove(temp_path.c_str());
  return true;
}

int main(int argc, char **argv) {
  CLI::App       app{"fx2-cmix - Advanced compression tool"};

  EPredictorType predictor_type = EPredictorType::FULL;

  // Global options
  unsigned int seed = 2;
  app.add_option("--seed,-s", seed, "Random seed for initialization (0-65535)")
      ->check(CLI::Range(0u, 65535u));

  int ppmd_mb    = 14000;
  int ppmd_order = 25;
  app.add_option("--ppmd-mb", ppmd_mb, "PPMD memory in MB (value << 20 bytes)")
      ->check(CLI::Range(1, 32768));
  app.add_option("--ppmd-order", ppmd_order, "PPMD model order")
      ->check(CLI::Range(1, 32));

  std::string predictor_name = "full";
  app.add_option("--predictor,-p", predictor_name,
                 "Predictor type: 'full' (default), 'ppmd', 'generic', 'lstm', 'bytemixer', or 'mix'")
      ->check(CLI::IsMember({"full", "ppmd", "generic", "lstm", "bytemixer", "mix"}));

    // Experimental: mixer weighting controls (apply to 'generic')
    bool  mw_enable = false;
    double mw_alpha = 0.001;
    double mw_min   = 0.05;
    double mw_max   = 1.5;
    app.add_flag("--mw-enable", mw_enable,
           "Enable adaptive weighting of layer-0 mixer outputs (generic)");
    app.add_option("--mw-alpha", mw_alpha, "EMA alpha for mixer weighting")
      ->check(CLI::Range(1e-6, 0.5));
    app.add_option("--mw-min", mw_min, "Minimum mixer weight")
      ->check(CLI::Range(0.0, 10.0));
    app.add_option("--mw-max", mw_max, "Maximum mixer weight")
      ->check(CLI::Range(0.0, 10.0));

  int lstm_bptt_depth = 0;
  app.add_option("--lstm-bptt-depth", lstm_bptt_depth,
                 "LSTM truncated BPTT depth (0 = full horizon, default: 0)")
      ->check(CLI::Range(0, 1024));

  int lstm_bptt_period = 1;
  app.add_option("--lstm-bptt-period", lstm_bptt_period,
                 "LSTM BPTT period (1 = every cycle, N = every Nth, default: 1)")
      ->check(CLI::Range(1, 128));

  int lstm_cells = 200;
  app.add_option("--lstm-num-cells", lstm_cells,
                 "LSTM hidden cells (default: 200, must be multiple of 8)")
      ->check(CLI::Range(16, 1024));

  int lstm_layers = 1;
  app.add_option("--lstm-num-layers", lstm_layers,
                 "LSTM layers (default: 1)")
      ->check(CLI::Range(1, 4));

  // Subcommands
  std::string input_path;
  std::string output_path;
  std::string dictionary_path;

  bool        compress_mode          = false;
  bool        decompress_mode        = false;
  bool        store_mode             = false;
  bool        no_preprocess_mode     = false;
  bool        enwik9_compress_mode   = false;
  bool        enwik9_decompress_mode = false;
  bool        extract_mode           = false;

  // Compress subcommand
  auto *compress =
      app.add_subcommand("compress", "Compress file with preprocessing");
  compress->add_option("input", input_path, "Input file path")->required();
  compress->add_option("output", output_path, "Output file path")->required();
  compress->add_option("-d,--dict", dictionary_path, "Dictionary file");
  compress->callback([&]() { compress_mode = true; });

  // Decompress subcommand
  auto *decompress = app.add_subcommand("decompress", "Decompress file");
  decompress->add_option("input", input_path, "Input file path")->required();
  decompress->add_option("output", output_path, "Output file path")->required();
  decompress->add_option("-d,--dict", dictionary_path, "Dictionary file");
  decompress->callback([&]() { decompress_mode = true; });

  // No-preprocess subcommand
  auto *no_preprocess =
      app.add_subcommand("no-preprocess", "Compress without preprocessing");
  no_preprocess->add_option("input", input_path, "Input file path")->required();
  no_preprocess->add_option("output", output_path, "Output file path")
      ->required();
  no_preprocess->callback([&]() { no_preprocess_mode = true; });

  // Store subcommand (only preprocessing)
  auto *store = app.add_subcommand("store", "Only apply preprocessing");
  store->add_option("input", input_path, "Input file path")->required();
  store->add_option("output", output_path, "Output file path")->required();
  store->add_option("-d,--dict", dictionary_path, "Dictionary file");
  store->callback([&]() { store_mode = true; });

  // Enwik9 compress subcommand
  auto *enwik9_compress =
      app.add_subcommand("enwik9-compress", "Compress enwik9 for Hutter Prize");
  enwik9_compress->add_option("input", input_path, "Input enwik9 file")
      ->required();
  enwik9_compress->add_option("output", output_path, "Output archive")
      ->required();
  enwik9_compress->callback([&]() { enwik9_compress_mode = true; });

  // Enwik9 decompress subcommand (no arguments needed)
  auto *enwik9_decompress = app.add_subcommand(
      "enwik9-decompress", "Decompress enwik9 Hutter Prize archive");
  enwik9_decompress->callback([&]() { enwik9_decompress_mode = true; });

  // Extract subcommand
  auto *extract =
      app.add_subcommand("extract", "Extract compressed file (internal use)");
  extract->add_option("input", input_path, "Input file path")->required();
  extract->add_option("output", output_path, "Output file path")->required();
  extract->callback([&]() { extract_mode = true; });

  // Header subcommand
  int   dict_size      = 0;
  int   new_order_size = 0;
  int   decomp_size    = 0;
  auto *header = app.add_subcommand("header", "Create header for Hutter Prize");
  header->add_option("dict_size", dict_size, "Compressed dictionary size")
      ->required();
  header
      ->add_option("new_order_size", new_order_size,
                   "Compressed new order size")
      ->required();
  header->add_option("decomp_size", decomp_size, "Decompressed input size")
      ->required();

  // Allow no subcommand for enwik9-decompress mode (backward compatibility)
  app.require_subcommand(0, 1);

  // Parse arguments
  CLI11_PARSE(app, argc, argv);

  // Handle no-arguments case (enwik9-decompress)
  if (app.get_subcommands().empty()) {
    enwik9_decompress_mode = true;
  }

  // Handle header subcommand
  if (header->parsed()) {
    HeaderInfo header_info;
    header_info.dict_size              = dict_size;
    header_info.new_article_order_size = new_order_size;
    header_info.decomp_input_size      = decomp_size;
    write("header.dat", header_info);
    return 0;
  }

  set_seed(seed);
  printf("Using random seed: %u\n", seed);

  // Parse predictor type
  if (predictor_name == "ppmd") {
    predictor_type = EPredictorType::PPMD_ONLY;
    printf("Using predictor: PPMD only\n");
  } else if (predictor_name == "generic") {
    predictor_type = EPredictorType::GENERIC;
    printf("Using predictor: Generic full predictor (explicit init)\n");
  } else if (predictor_name == "lstm") {
    predictor_type = EPredictorType::LSTM_ONLY;
    printf("Using predictor: LSTM only (%d cells, %d layers)\n", lstm_cells, lstm_layers);
  } else if (predictor_name == "bytemixer") {
    predictor_type = EPredictorType::BYTE_MIXER;
    printf("Using predictor: ByteMixer (PPMd + LSTM, %d cells, %d layers)\n", lstm_cells, lstm_layers);
  } else if (predictor_name == "mix") {
    predictor_type = EPredictorType::MIX;
    printf("Using predictor: Mix (PPMd + LSTM + Bracket, %d cells, %d layers)\n", lstm_cells, lstm_layers);
  } else {
    predictor_type = EPredictorType::FULL;
    printf("Using predictor: Full (all models)\n");
  }
  if (lstm_bptt_depth > 0) {
    printf("Using LSTM BPTT depth: %d\n", lstm_bptt_depth);
  }
  if (lstm_bptt_period > 1) {
    printf("Using LSTM BPTT period: %d\n", lstm_bptt_period);
  }
  if (lstm_cells % 8 != 0) {
    int suggested = (lstm_cells + 7) & ~7;
    printf("WARNING: --lstm-num-cells=%d is not a multiple of 8. "
           "weights_ column stride = %d bytes, not 32-byte aligned for AVX2. "
           "Suggested value: %d\n", lstm_cells, lstm_cells * 4, suggested);
  }

  clock_t start             = clock();

  bool    enable_preprocess = !no_preprocess_mode;
  FILE   *dictionary        = nullptr;

  if (!dictionary_path.empty()) {
    dictionary = nullptr;
    if (fopen_s(&dictionary, dictionary_path.c_str(), "rb") != 0 || !dictionary) {
      fprintf(stderr, "Error: Cannot open dictionary file: %s\n",
              dictionary_path.c_str());
      return 1;
    }
  }

  std::string        temp_path   = output_path + ".cmix.temp";
  unsigned long long input_bytes = 0, output_bytes = 0;

  if (enwik9_decompress_mode) {
    // Decompress enwik9
    //  unpack a) header b) cmix dictionary, c) new order of articles, d) actual
    //  cmix binary
    selfextract_decomp();

    // run compression
    std::cout << "Running cmix decompression..." << std::endl;
    input_path  = ".ready4cmix_decomp";
    output_path = ".input_decomp";
    dictionary  = nullptr;
    if (fopen_s(&dictionary, ".dict", "rb") != 0 || !dictionary) {
      fprintf(stderr, "Error: Cannot open dictionary file for enwik9 decompress: %s\n",
              ".dict");
      abort();
    }

    if (!RunDecompression(predictor_type, input_path, temp_path, output_path,
                dictionary, &input_bytes, &output_bytes, ppmd_order,
                ppmd_mb, mw_enable, mw_alpha, mw_min, mw_max,
                lstm_bptt_depth, lstm_bptt_period,
                lstm_cells, lstm_layers)) {
      fprintf(stderr, "Error: Enwik9 decompression failed\n");
      return 1;
    }
    std::cout << "Cmix decompression finished" << std::endl;

    split4Decomp();

    // apply phda9 preprocessor
    phda9_resto();

    // change the order of articles in the input
    sort();

    // merge all input parts after preprocessing
    cat(".intro_decomp", ".main_decomp_restored_sorted", "un1_d");
    cat("un1_d", ".coda_decomp", "enwik9_uncompressed");

    goto print_end_message;
  }

  // Handle store mode
  if (store_mode) {
    if (!Store(input_path, temp_path, output_path, dictionary, &input_bytes,
               &output_bytes)) {
      fprintf(stderr, "Error: Store operation failed\n");
      return 1;
    }
  }

  // Handle compress/no-preprocess mode
  else if (compress_mode || no_preprocess_mode) {
    remove(".dict");
    if (!RunCompression(predictor_type, enable_preprocess, input_path,
              temp_path, output_path, dictionary, &input_bytes,
              &output_bytes, ppmd_order, ppmd_mb, mw_enable, mw_alpha, mw_min, mw_max,
              lstm_bptt_depth, lstm_bptt_period,
              lstm_cells, lstm_layers)) {
      fprintf(stderr, "Error: Compression failed\n");
      return 1;
    }
  }

  // Handle enwik9 compress mode
  else if (enwik9_compress_mode) {
    // unpack a) cmix dictionary, b) new order of articles, c) actual cmix
    // binary
    selfextract_comp();

    // Preparing enwik9 for reordering
    split4Comp(input_path.c_str());

    // change the order of articles in the input
    reorder();

    // apply phda9 preprocessor
    phda9_prepr();

    // merge all input parts after preprocessing
    cat(".main_phda9prepr", ".intro", "un1");
    cat("un1", ".coda", ".ready4cmix");

    // run compression
    std::string orig_input = input_path;
    input_path             = ".ready4cmix";
    temp_path              = output_path + ".cmix.temp";
    dictionary             = nullptr;
    if (fopen_s(&dictionary, ".dict", "rb") != 0 || !dictionary) {
      fprintf(stderr, "Error: Cannot open dictionary file for enwik9 compress: %s\n",
              ".dict");
      abort();
    }
    if (!RunCompression(predictor_type, enable_preprocess, input_path,
              temp_path, output_path, dictionary, &input_bytes,
              &output_bytes, ppmd_order, ppmd_mb, mw_enable, mw_alpha, mw_min, mw_max,
              lstm_bptt_depth, lstm_bptt_period,
              lstm_cells, lstm_layers)) {
      fprintf(stderr, "Error: Enwik9 compression failed\n");
      return 1;
    }

    // construct a selfextracting decompressor binary
    // archive9 = decomp_binary(upxed) + comp_dict + cmix_output + header.dat
    cat(".decomp_bin", ".dict.comp", "dec1");

    // get the size of the output file
    size_t     output_size = getFileSize(output_path);

    HeaderInfo header_info;
    read("test.dat", header_info);
    header_info.decomp_input_size = output_size;
    write("header4archive.dat", header_info);

    cat("dec1", output_path.c_str(), "dec2");
    cat("dec2", "header4archive.dat", "archive9");

    // make the decompressor binary executable (Unix only)
#ifndef _WIN32
    char mode[]   = "0777";
    char buf[100] = "archive9";
    int  i        = strtol(mode, 0, 8);
    chmod(buf, i);
#endif
  }

  // Handle extract mode
  else if (extract_mode) {
    dictionary = nullptr;
    if (fopen_s(&dictionary, ".dict", "rb") != 0 || !dictionary) {
      fprintf(stderr, "Error: Cannot open dictionary file for extract: %s\n",
              ".dict");
      abort();
    }
    if (!RunDecompression(predictor_type, input_path, temp_path, output_path,
                dictionary, &input_bytes, &output_bytes, ppmd_order,
                ppmd_mb, mw_enable, mw_alpha, mw_min, mw_max,
                lstm_bptt_depth, lstm_bptt_period,
                lstm_cells, lstm_layers)) {
      fprintf(stderr, "Error: Extract operation failed\n");
      return 1;
    }
    goto print_end_message;
  }

  // Handle decompress mode
  else if (decompress_mode) {
    if (!RunDecompression(predictor_type, input_path, temp_path, output_path,
                dictionary, &input_bytes, &output_bytes, ppmd_order,
                ppmd_mb, mw_enable, mw_alpha, mw_min, mw_max,
                lstm_bptt_depth, lstm_bptt_period,
                lstm_cells, lstm_layers)) {
      fprintf(stderr, "Error: Decompression failed\n");
      return 1;
    }
  }

print_end_message:
  double seconds             = ((double)clock() - start) / CLOCKS_PER_SEC;
  double bits_per_second     = (8.0 * (double)input_bytes) / (seconds);
  double nanoseconds_per_bit = (seconds * 1e9) / (8.0 * (double)input_bytes);
  double compression_ratio = 100.0 * (double)output_bytes / (double)input_bytes;
  double comp_factor = (double)input_bytes / (double)output_bytes;
  printf(
      "\r%lld bytes -> %lld bytes in %1.2f s.\nSpeed: %1.2f bits/s (%1.2f "
      "bytes/s). %1.2f ns/bit\nCompression ratio: %1.5f%%\nCompr.Factor:      %1.5f\n",
      input_bytes, output_bytes, seconds, bits_per_second,
      bits_per_second / 8.0, nanoseconds_per_bit, compression_ratio, comp_factor);

  return 0;
}
