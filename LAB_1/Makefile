CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra

all: matrix_seq

matrix_seq: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o matrix_seq

clean:
	rm -f matrix_seq
