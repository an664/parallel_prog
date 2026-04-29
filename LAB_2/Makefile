CXX = g++-15
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp

all: matrix_omp

matrix_omp: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o matrix_omp

clean:
	rm -f matrix_omp
