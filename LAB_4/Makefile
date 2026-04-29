CXX = clang++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
LDFLAGS ?= -framework OpenCL

all: matrix_opencl

matrix_opencl: matrix_opencl.cpp
	$(CXX) $(CXXFLAGS) matrix_opencl.cpp $(LDFLAGS) -o matrix_opencl

clean:
	rm -f matrix_opencl
