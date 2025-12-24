CC = clang++

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

CPPFLAGS_PART-THAT-SHOULD-BE-FAST := $(CFLAGS_DEFINES) -DUPDATE_LIMIT=3000 -m64 -Wall -std=c++17 -fno-threadsafe-statics -Wunknown-pragmas -Wno-unused-variable -fno-threadsafe-statics -Wno-unused-but-set-variable -Wno-format -Ithird_party -DEIGEN_NO_DEBUG -DEIGEN_FAST_MATH=1 -DEIGEN_UNROLLING_LIMIT=100 -DNDEBUG

ifdef COREI7
$(info COREI7 defined)
CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -march=corei7
else
ifdef ZEN2
$(info ZEN2 defined)
CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -march=znver2
else
CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -march=native -mtune=native
$(info native used)
endif
endif

# Profile build flags (full optimization + debug symbols for profiling tools like VTune)
CPPFLAGS_PROFILE_SLOW  := $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST)
CPPFLAGS_PROFILE_FAST  := $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST)
CPPFLAGS_PROFILE_SLOW  += -g -Os -fdata-sections -ffunction-sections
CPPFLAGS_PROFILE_FAST  += -g -O3 -ffast-math -fhonor-nans -fhonor-infinities -fno-finite-math-only -fvectorize -fslp-vectorize -funroll-loops -fdata-sections -ffunction-sections
LFLAGS_PROFILE         := -m64 -std=c++17 -g

# Production build flags (full optimization, LTO)
CPPFLAGS_PART-THAT-CAN-BE-SLOW    := $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST)
CPPFLAGS_PART-THAT-CAN-BE-SLOW    += -Os -fdata-sections -ffunction-sections -flto
CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -O3 -ffast-math -fno-finite-math-only -fvectorize -fslp-vectorize -funroll-loops -fdata-sections -ffunction-sections -flto -fomit-frame-pointer -falign-functions=32
LFLAGS := -m64 -std=c++17 -flto -fuse-ld=lld

OUT_DIR := out

# Create output directory
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/preprocess_chunked:
	mkdir -p $(OUT_DIR)/preprocess_chunked

# Chunked v2 experimental tools
CHUNKED_DIR := src/preprocess_chunked_v2
CHUNKED_OBJS := \
	$(OUT_DIR)/preprocess_chunked/chunked_preprocessor_v2.o \
	$(OUT_DIR)/preprocess_chunked/numbers_codec.o \
	$(OUT_DIR)/preprocess_chunked/text_utils.o

$(OUT_DIR)/preprocess_chunked/text_utils.o: $(CHUNKED_DIR)/text_utils.cpp $(CHUNKED_DIR)/text_utils.hpp | $(OUT_DIR)/preprocess_chunked
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c $(CHUNKED_DIR)/text_utils.cpp -o $(OUT_DIR)/preprocess_chunked/text_utils.o

$(OUT_DIR)/preprocess_chunked/chunked_preprocessor_v2.o: $(CHUNKED_DIR)/chunked_preprocessor_v2.cpp $(CHUNKED_DIR)/chunked_preprocessor_v2.hpp | $(OUT_DIR)/preprocess_chunked
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c $(CHUNKED_DIR)/chunked_preprocessor_v2.cpp -o $(OUT_DIR)/preprocess_chunked/chunked_preprocessor_v2.o

$(OUT_DIR)/preprocess_chunked/numbers_codec.o: $(CHUNKED_DIR)/numbers_codec.cpp $(CHUNKED_DIR)/numbers_codec.hpp | $(OUT_DIR)/preprocess_chunked
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c $(CHUNKED_DIR)/numbers_codec.cpp -o $(OUT_DIR)/preprocess_chunked/numbers_codec.o

$(OUT_DIR)/preprocess_chunked/main_chunked.o: $(CHUNKED_DIR)/main.cpp $(CHUNKED_DIR)/chunked_preprocessor_v2.hpp $(CHUNKED_DIR)/numbers_codec.hpp | $(OUT_DIR)/preprocess_chunked
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c $(CHUNKED_DIR)/main.cpp -o $(OUT_DIR)/preprocess_chunked/main_chunked.o

$(OUT_DIR)/preprocess_chunked/stats_v2.o: $(CHUNKED_DIR)/stats_v2.cpp $(CHUNKED_DIR)/numbers_codec.hpp | $(OUT_DIR)/preprocess_chunked
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c $(CHUNKED_DIR)/stats_v2.cpp -o $(OUT_DIR)/preprocess_chunked/stats_v2.o

$(OUT_DIR)/preprocess_chunked/dump_v2.o: $(CHUNKED_DIR)/dump_v2.cpp $(CHUNKED_DIR)/numbers_codec.hpp | $(OUT_DIR)/preprocess_chunked
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c $(CHUNKED_DIR)/dump_v2.cpp -o $(OUT_DIR)/preprocess_chunked/dump_v2.o

chunked-tools: $(CHUNKED_OBJS) $(OUT_DIR)/preprocess_chunked/main_chunked.o $(OUT_DIR)/preprocess_chunked/stats_v2.o $(OUT_DIR)/preprocess_chunked/dump_v2.o
	$(CC) $(LFLAGS) $(CHUNKED_OBJS) $(OUT_DIR)/preprocess_chunked/main_chunked.o -o chunked_main.exe
	$(CC) $(LFLAGS) $(CHUNKED_OBJS) $(OUT_DIR)/preprocess_chunked/stats_v2.o -o stats_v2.exe
	$(CC) $(LFLAGS) $(CHUNKED_OBJS) $(OUT_DIR)/preprocess_chunked/dump_v2.o -o dump_v2.exe

prof_gen: CPPFLAGS_PART-THAT-CAN-BE-SLOW    += -fprofile-generate=$(ROOT_DIR)/pgo_data
prof_gen: CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -fprofile-generate=$(ROOT_DIR)/pgo_data
prof_gen: LFLAGS                            += -fprofile-generate=$(ROOT_DIR)/pgo_data
prof_gen: clean cmix

prof_use: CPPFLAGS_PART-THAT-CAN-BE-SLOW    += -fprofile-use=$(ROOT_DIR)/pgo_data 
prof_use: CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -fprofile-use=$(ROOT_DIR)/pgo_data 
prof_use: LFLAGS                            += -fprofile-use=$(ROOT_DIR)/pgo_data 
prof_use: clean cmix


# Object files with slow optimization
$(OUT_DIR)/preprocessor.o: src/preprocess/preprocessor.cpp src/preprocess/preprocessor.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-CAN-BE-SLOW) -c src/preprocess/preprocessor.cpp -o $(OUT_DIR)/preprocessor.o

$(OUT_DIR)/dictionary.o: src/preprocess/dictionary.cpp src/preprocess/dictionary.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-CAN-BE-SLOW) -c src/preprocess/dictionary.cpp -o $(OUT_DIR)/dictionary.o

# Object files with fast optimization
$(OUT_DIR)/decoder.o: src/coder/decoder.cpp src/coder/decoder.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/coder/decoder.cpp -o $(OUT_DIR)/decoder.o

$(OUT_DIR)/encoder.o: src/coder/encoder.cpp src/coder/encoder.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/coder/encoder.cpp -o $(OUT_DIR)/encoder.o

$(OUT_DIR)/context-manager.o: src/context-manager.cpp src/context-manager.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/context-manager.cpp -o $(OUT_DIR)/context-manager.o

$(OUT_DIR)/bit-context.o: src/contexts/bit-context.cpp src/contexts/bit-context.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/contexts/bit-context.cpp -o $(OUT_DIR)/bit-context.o

$(OUT_DIR)/bracket-context.o: src/contexts/bracket-context.cpp src/contexts/bracket-context.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/contexts/bracket-context.cpp -o $(OUT_DIR)/bracket-context.o

$(OUT_DIR)/context-hash.o: src/contexts/context-hash.cpp src/contexts/context-hash.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/contexts/context-hash.cpp -o $(OUT_DIR)/context-hash.o

$(OUT_DIR)/indirect-hash.o: src/contexts/indirect-hash.cpp src/contexts/indirect-hash.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/contexts/indirect-hash.cpp -o $(OUT_DIR)/indirect-hash.o

$(OUT_DIR)/interval-hash.o: src/contexts/interval-hash.cpp src/contexts/interval-hash.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/contexts/interval-hash.cpp -o $(OUT_DIR)/interval-hash.o

$(OUT_DIR)/interval.o: src/contexts/interval.cpp src/contexts/interval.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/contexts/interval.cpp -o $(OUT_DIR)/interval.o

$(OUT_DIR)/sparse.o: src/contexts/sparse.cpp src/contexts/sparse.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/contexts/sparse.cpp -o $(OUT_DIR)/sparse.o

$(OUT_DIR)/bracket.o: src/models/bracket.cpp src/models/bracket.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/models/bracket.cpp -o $(OUT_DIR)/bracket.o

$(OUT_DIR)/preprocess_chunked/dump_chunked.o: src/preprocess_chunked/dump_chunked.cpp | $(OUT_DIR)/preprocess_chunked
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/preprocess_chunked/dump_chunked.cpp -o $(OUT_DIR)/preprocess_chunked/dump_chunked.o

$(OUT_DIR)/byte-model.o: src/models/byte-model.cpp src/models/byte-model.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/models/byte-model.cpp -o $(OUT_DIR)/byte-model.o

$(OUT_DIR)/direct-hash.o: src/models/direct-hash.cpp src/models/direct-hash.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/models/direct-hash.cpp -o $(OUT_DIR)/direct-hash.o

$(OUT_DIR)/direct.o: src/models/direct.cpp src/models/direct.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/models/direct.cpp -o $(OUT_DIR)/direct.o

$(OUT_DIR)/indirect.o: src/models/indirect.cpp src/models/indirect.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/models/indirect.cpp -o $(OUT_DIR)/indirect.o

$(OUT_DIR)/match.o: src/models/match.cpp src/models/match.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/models/match.cpp -o $(OUT_DIR)/match.o

$(OUT_DIR)/fxcmv1.o: src/models/fxcmv1.cpp src/models/fxcmv1.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/models/fxcmv1.cpp -o $(OUT_DIR)/fxcmv1.o

$(OUT_DIR)/ppmd.o: src/models/ppmd.cpp src/models/ppmd.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/models/ppmd.cpp -o $(OUT_DIR)/ppmd.o

$(OUT_DIR)/url-model.o: src/models/url-model.cpp src/models/url-model.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/models/url-model.cpp -o $(OUT_DIR)/url-model.o

$(OUT_DIR)/nonstationary.o: src/states/nonstationary.cpp src/states/nonstationary.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/states/nonstationary.cpp -o $(OUT_DIR)/nonstationary.o

$(OUT_DIR)/run-map.o: src/states/run-map.cpp src/states/run-map.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/states/run-map.cpp -o $(OUT_DIR)/run-map.o

$(OUT_DIR)/byte-mixer.o: src/mixer/byte-mixer.cpp src/mixer/byte-mixer.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/mixer/byte-mixer.cpp -o $(OUT_DIR)/byte-mixer.o

$(OUT_DIR)/mixer-input.o: src/mixer/mixer-input.cpp src/mixer/mixer-input.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/mixer/mixer-input.cpp -o $(OUT_DIR)/mixer-input.o

$(OUT_DIR)/mixer.o: src/mixer/mixer.cpp src/mixer/mixer.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/mixer/mixer.cpp -o $(OUT_DIR)/mixer.o

$(OUT_DIR)/sigmoid.o: src/mixer/sigmoid.cpp src/mixer/sigmoid.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/mixer/sigmoid.cpp -o $(OUT_DIR)/sigmoid.o

$(OUT_DIR)/sse.o: src/mixer/sse.cpp src/mixer/sse.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/mixer/sse.cpp -o $(OUT_DIR)/sse.o

$(OUT_DIR)/predictor.o: src/predictor.cpp src/predictor.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/predictor.cpp -o $(OUT_DIR)/predictor.o

$(OUT_DIR)/random.o: src/random.cpp | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/random.cpp -o $(OUT_DIR)/random.o

$(OUT_DIR)/runner.o: src/runner.cpp | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/runner.cpp -o $(OUT_DIR)/runner.o

$(OUT_DIR)/generic_full_predictor.o: src/generic_full_predictor.cpp src/generic_full_predictor.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/generic_full_predictor.cpp -o $(OUT_DIR)/generic_full_predictor.o

slow: $(OUT_DIR)/preprocessor.o $(OUT_DIR)/dictionary.o

fast: $(OUT_DIR)/decoder.o $(OUT_DIR)/encoder.o $(OUT_DIR)/random.o $(OUT_DIR)/context-manager.o $(OUT_DIR)/bit-context.o $(OUT_DIR)/bracket-context.o $(OUT_DIR)/context-hash.o $(OUT_DIR)/indirect-hash.o $(OUT_DIR)/interval-hash.o $(OUT_DIR)/interval.o $(OUT_DIR)/sparse.o $(OUT_DIR)/bracket.o $(OUT_DIR)/byte-model.o $(OUT_DIR)/direct-hash.o $(OUT_DIR)/direct.o $(OUT_DIR)/indirect.o $(OUT_DIR)/match.o $(OUT_DIR)/fxcmv1.o $(OUT_DIR)/ppmd.o $(OUT_DIR)/url-model.o $(OUT_DIR)/nonstationary.o $(OUT_DIR)/run-map.o $(OUT_DIR)/byte-mixer.o $(OUT_DIR)/mixer-input.o $(OUT_DIR)/mixer.o $(OUT_DIR)/sigmoid.o $(OUT_DIR)/sse.o $(OUT_DIR)/predictor.o $(OUT_DIR)/generic_full_predictor.o $(OUT_DIR)/runner.o

# Production build (default: full optimizations + LTO)
# Ensure even traditionally "slow" objects are compiled with fast flags for cmix
cmix-prod: CPPFLAGS_PART-THAT-CAN-BE-SLOW := $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST)
cmix-prod: fast slow
	$(CC) $(LFLAGS) $(OUT_DIR)/bit-context.o $(OUT_DIR)/random.o $(OUT_DIR)/bracket-context.o $(OUT_DIR)/bracket.o $(OUT_DIR)/byte-mixer.o $(OUT_DIR)/byte-model.o $(OUT_DIR)/context-hash.o $(OUT_DIR)/context-manager.o $(OUT_DIR)/decoder.o $(OUT_DIR)/dictionary.o $(OUT_DIR)/direct-hash.o $(OUT_DIR)/direct.o $(OUT_DIR)/encoder.o $(OUT_DIR)/indirect-hash.o $(OUT_DIR)/indirect.o $(OUT_DIR)/interval-hash.o $(OUT_DIR)/interval.o $(OUT_DIR)/match.o $(OUT_DIR)/mixer-input.o $(OUT_DIR)/mixer.o $(OUT_DIR)/nonstationary.o $(OUT_DIR)/fxcmv1.o $(OUT_DIR)/ppmd.o $(OUT_DIR)/url-model.o $(OUT_DIR)/predictor.o $(OUT_DIR)/generic_full_predictor.o $(OUT_DIR)/preprocessor.o $(OUT_DIR)/run-map.o $(OUT_DIR)/runner.o $(OUT_DIR)/sigmoid.o $(OUT_DIR)/sparse.o $(OUT_DIR)/sse.o -o cmix.exe

# Profile build (full optimizations + debug symbols for VTune/perf/gprof)
cmix-profile: CPPFLAGS_PART-THAT-CAN-BE-SLOW := $(CPPFLAGS_PROFILE_SLOW)
cmix-profile: CPPFLAGS_PART-THAT-SHOULD-BE-FAST := $(CPPFLAGS_PROFILE_FAST)
cmix-profile: LFLAGS := $(LFLAGS_PROFILE)
cmix-profile: clean fast slow
	$(CC) $(LFLAGS_PROFILE) $(OUT_DIR)/bit-context.o $(OUT_DIR)/random.o $(OUT_DIR)/bracket-context.o $(OUT_DIR)/bracket.o $(OUT_DIR)/byte-mixer.o $(OUT_DIR)/byte-model.o $(OUT_DIR)/context-hash.o $(OUT_DIR)/context-manager.o $(OUT_DIR)/decoder.o $(OUT_DIR)/dictionary.o $(OUT_DIR)/direct-hash.o $(OUT_DIR)/direct.o $(OUT_DIR)/encoder.o $(OUT_DIR)/indirect-hash.o $(OUT_DIR)/indirect.o $(OUT_DIR)/interval-hash.o $(OUT_DIR)/interval.o $(OUT_DIR)/match.o $(OUT_DIR)/mixer-input.o $(OUT_DIR)/mixer.o $(OUT_DIR)/nonstationary.o $(OUT_DIR)/fxcmv1.o $(OUT_DIR)/ppmd.o $(OUT_DIR)/url-model.o $(OUT_DIR)/predictor.o $(OUT_DIR)/generic_full_predictor.o $(OUT_DIR)/preprocessor.o $(OUT_DIR)/run-map.o $(OUT_DIR)/runner.o $(OUT_DIR)/sigmoid.o $(OUT_DIR)/sparse.o $(OUT_DIR)/sse.o -o cmix.exe

# Keep old 'cmix' target for backward compatibility (defaults to production)
cmix: cmix-prod

remap: src/readalike_prepr/article_remap.cpp
	$(CC) src/readalike_prepr/article_remap.cpp -o remap.exe

# PPMD isolated performance test (with debug symbols)
$(OUT_DIR)/ppmd-debug.o: src/models/ppmd.cpp src/models/ppmd.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -g -c src/models/ppmd.cpp -o $(OUT_DIR)/ppmd-debug.o

$(OUT_DIR)/byte-model-debug.o: src/models/byte-model.cpp src/models/byte-model.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -g -c src/models/byte-model.cpp -o $(OUT_DIR)/byte-model-debug.o

test-ppmd: $(OUT_DIR)/ppmd-debug.o $(OUT_DIR)/byte-model-debug.o
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -g -fuse-ld=lld src/test_ppmd.cpp $(OUT_DIR)/ppmd-debug.o $(OUT_DIR)/byte-model-debug.o -o test-ppmd.exe

# PPMD memory diagnostic test (direct ppmd_Model testing - includes ppmd.cpp directly)
# Note: Removes -DNDEBUG to enable assert() checks for debugging
test-ppmd-mem: $(OUT_DIR)/byte-model-debug.o
	$(CC) $(filter-out -DNDEBUG,$(CPPFLAGS_PART-THAT-SHOULD-BE-FAST)) -g -fuse-ld=lld src/test_ppmd_mem.cpp $(OUT_DIR)/byte-model-debug.o -o test-ppmd-mem.exe

# UTF-8 character statistics analyzer
test-utf8:
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -g -fuse-ld=lld src/test_utf8.cpp -o test-utf8.exe

# Longest words analyzer (UTF-8, trie-based)
test-longest-words:
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -g -fuse-ld=lld src/test_longest_words.cpp -o test-longest-words.exe

# UTF-8 preprocessor test (compress/decompress with word remapping)
$(OUT_DIR)/utf8_preprocessor.o: src/preprocess_utf8/utf8_preprocessor.cpp src/preprocess_utf8/utf8_preprocessor.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/preprocess_utf8/utf8_preprocessor.cpp -o $(OUT_DIR)/utf8_preprocessor.o

test-utf8-preprocess: $(OUT_DIR)/utf8_preprocessor.o
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -g -fuse-ld=lld src/test_utf8_preprocess.cpp $(OUT_DIR)/utf8_preprocessor.o -o test-utf8-preprocess.exe

# Chunked preprocessor test (region-based preprocessing)
$(OUT_DIR)/chunked_preprocessor.o: src/preprocess_chunked/chunked_preprocessor.cpp src/preprocess_chunked/chunked_preprocessor.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/preprocess_chunked/chunked_preprocessor.cpp -o $(OUT_DIR)/chunked_preprocessor.o

test-chunked-preprocess: $(OUT_DIR)/chunked_preprocessor.o
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -g -fuse-ld=lld src/preprocess_chunked/test_chunked_preprocess.cpp $(OUT_DIR)/chunked_preprocessor.o -o test-chunked-preprocess.exe

dump-chunked.exe: $(OUT_DIR)/preprocess_chunked/dump_chunked.o
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -g -fuse-ld=lld $^ -o dump-chunked.exe

$(OUT_DIR)/byte-model.o: src/models/byte-model.cpp src/models/byte-model.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/models/byte-model.cpp -o $(OUT_DIR)/byte-model.o

# V2 stats tool (chunked preprocessor version 2)
stats_v2.exe: src/preprocess_chunked_v2/stats_v2.cpp
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -g -fuse-ld=lld $< -o $@

# URL model test tool
test-url-model.exe: src/test-url-model.cpp $(OUT_DIR)/url-model.o $(OUT_DIR)/byte-model.o $(OUT_DIR)/ppmd.o | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -fuse-ld=lld $^ -o $@

clean:
	rm -rf $(OUT_DIR)
	rm -f cmix.exe
	rm -f remap.exe
	rm -f test-ppmd.exe
	rm -f test-utf8.exe
	rm -f test-longest-words.exe
	rm -f test-utf8-preprocess.exe
	rm -f test-chunked-preprocess.exe
	rm -f dump-chunked.exe
	rm -f test-url-model.exe

all: cmix remap

# Dictionary-based compression prototype
dictionary-compress:
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -Ithird_party -fuse-ld=lld src/dictionary_compress.cpp -o dictionary-compress.exe

