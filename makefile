CC = c++

ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

CPPFLAGS_PART-THAT-SHOULD-BE-FAST := $(CFLAGS_DEFINES) -DUPDATE_LIMIT=3000 -m64 -Wall -std=c++17 -fno-exceptions -fno-threadsafe-statics -Wunknown-pragmas -Wno-unused-variable -fno-exceptions -fno-threadsafe-statics -Wno-unused-but-set-variable -Wno-format -Ithird_party -DEIGEN_NO_DEBUG -DNDEBUG

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

CPPFLAGS_PART-THAT-CAN-BE-SLOW    := $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST)
CPPFLAGS_PART-THAT-CAN-BE-SLOW    += -Os -fdata-sections -ffunction-sections
CPPFLAGS_PART-THAT-SHOULD-BE-FAST += -O3 -ffast-math -fdata-sections -ffunction-sections
LFLAGS := -m64 -Wl,--gc-sections -std=c++17

OUT_DIR := out

# Create output directory
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

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

$(OUT_DIR)/combined-context.o: src/contexts/combined-context.cpp src/contexts/combined-context.h | $(OUT_DIR)
	$(CC) $(CPPFLAGS_PART-THAT-SHOULD-BE-FAST) -c src/contexts/combined-context.cpp -o $(OUT_DIR)/combined-context.o

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

slow: $(OUT_DIR)/preprocessor.o $(OUT_DIR)/dictionary.o

fast: $(OUT_DIR)/decoder.o $(OUT_DIR)/encoder.o $(OUT_DIR)/random.o $(OUT_DIR)/context-manager.o $(OUT_DIR)/bit-context.o $(OUT_DIR)/bracket-context.o $(OUT_DIR)/combined-context.o $(OUT_DIR)/context-hash.o $(OUT_DIR)/indirect-hash.o $(OUT_DIR)/interval-hash.o $(OUT_DIR)/interval.o $(OUT_DIR)/sparse.o $(OUT_DIR)/bracket.o $(OUT_DIR)/byte-model.o $(OUT_DIR)/direct-hash.o $(OUT_DIR)/direct.o $(OUT_DIR)/indirect.o $(OUT_DIR)/match.o $(OUT_DIR)/fxcmv1.o $(OUT_DIR)/ppmd.o $(OUT_DIR)/nonstationary.o $(OUT_DIR)/run-map.o $(OUT_DIR)/byte-mixer.o $(OUT_DIR)/mixer-input.o $(OUT_DIR)/mixer.o $(OUT_DIR)/sigmoid.o $(OUT_DIR)/sse.o $(OUT_DIR)/predictor.o $(OUT_DIR)/runner.o

cmix: fast slow
	$(CC) $(LFLAGS) $(OUT_DIR)/bit-context.o $(OUT_DIR)/random.o $(OUT_DIR)/bracket-context.o $(OUT_DIR)/bracket.o $(OUT_DIR)/byte-mixer.o $(OUT_DIR)/byte-model.o $(OUT_DIR)/combined-context.o $(OUT_DIR)/context-hash.o $(OUT_DIR)/context-manager.o $(OUT_DIR)/decoder.o $(OUT_DIR)/dictionary.o $(OUT_DIR)/direct-hash.o $(OUT_DIR)/direct.o $(OUT_DIR)/encoder.o $(OUT_DIR)/indirect-hash.o $(OUT_DIR)/indirect.o $(OUT_DIR)/interval-hash.o $(OUT_DIR)/interval.o $(OUT_DIR)/match.o $(OUT_DIR)/mixer-input.o $(OUT_DIR)/mixer.o $(OUT_DIR)/nonstationary.o $(OUT_DIR)/fxcmv1.o $(OUT_DIR)/ppmd.o $(OUT_DIR)/predictor.o $(OUT_DIR)/preprocessor.o $(OUT_DIR)/run-map.o $(OUT_DIR)/runner.o $(OUT_DIR)/sigmoid.o $(OUT_DIR)/sparse.o $(OUT_DIR)/sse.o -o cmix

remap: src/readalike_prepr/article_remap.cpp
	$(CC) src/readalike_prepr/article_remap.cpp -o remap

clean:
	rm -rf $(OUT_DIR)
	rm -f cmix
	rm -f remap

all: cmix remap

