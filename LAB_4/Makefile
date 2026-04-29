NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17

all: matrix_cuda

matrix_cuda: matrix_cuda.cu
	$(NVCC) $(NVCCFLAGS) matrix_cuda.cu -o matrix_cuda

clean:
	rm -f matrix_cuda
