CXX = clang++
CXXFLAGS = -std=c++17 -O3 -Wall -I..

SRC = chunked_preprocessor_v2.cpp bitstream_vector.cpp bitstream_file.cpp text_utils.cpp main.cpp dump_v2.cpp stats_v2.cpp
OBJ = $(SRC:.cpp=.o)

all: chunked_v2 dump_v2 stats_v2

chunked_v2: chunked_preprocessor_v2.o bitstream_vector.o bitstream_file.o text_utils.o main.o
	$(CXX) $(CXXFLAGS) $^ -o chunked_v2.exe

dump_v2: dump_v2.o bitstream_vector.o bitstream_file.o text_utils.o
	$(CXX) $(CXXFLAGS) $^ -o dump_v2.exe

stats_v2: stats_v2.o bitstream_vector.o bitstream_file.o text_utils.o
	$(CXX) $(CXXFLAGS) $^ -o stats_v2.exe

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	del *.o *.exe
