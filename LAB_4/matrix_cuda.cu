#include <cuda_runtime.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Matrix = std::vector<double>;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t error = (call);                                                   \
        if (error != cudaSuccess) {                                                   \
            throw std::runtime_error(std::string("CUDA error: ") +                   \
                                     cudaGetErrorString(error));                      \
        }                                                                             \
    } while (0)

Matrix read_matrix(const std::string& path, int& n) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open input file: " + path);
    }

    in >> n;
    if (n <= 0) {
        throw std::runtime_error("Matrix size must be positive");
    }

    Matrix matrix(static_cast<std::size_t>(n) * n);
    for (double& value : matrix) {
        if (!(in >> value)) {
            throw std::runtime_error("Not enough matrix values in: " + path);
        }
    }
    return matrix;
}

void write_matrix(const std::string& path, const Matrix& matrix, int n) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot open output file: " + path);
    }

    out << n << '\n' << std::setprecision(12);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << matrix[static_cast<std::size_t>(i) * n + j];
            out << (j + 1 == n ? '\n' : ' ');
        }
    }
}

__global__ void matmul_naive(const double* a, const double* b, double* c, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }

    double sum = 0.0;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

__global__ void matmul_tiled(const double* a, const double* b, double* c, int n) {
    extern __shared__ double shared[];
    double* tile_a = shared;
    double* tile_b = shared + blockDim.x * blockDim.y;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_size = blockDim.x;
    double sum = 0.0;

    for (int offset = 0; offset < n; offset += tile_size) {
        const int a_col = offset + threadIdx.x;
        const int b_row = offset + threadIdx.y;

        tile_a[threadIdx.y * tile_size + threadIdx.x] =
            (row < n && a_col < n) ? a[row * n + a_col] : 0.0;
        tile_b[threadIdx.y * tile_size + threadIdx.x] =
            (b_row < n && col < n) ? b[b_row * n + col] : 0.0;

        __syncthreads();

        for (int k = 0; k < tile_size; ++k) {
            sum += tile_a[threadIdx.y * tile_size + k] * tile_b[k * tile_size + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix_a.txt> <matrix_b.txt> <result.txt> <block_size> <naive|tiled>\n";
        return 1;
    }

    try {
        int n_a = 0;
        int n_b = 0;
        const Matrix a = read_matrix(argv[1], n_a);
        const Matrix b = read_matrix(argv[2], n_b);
        if (n_a != n_b) {
            throw std::runtime_error("Matrices must have the same size");
        }

        const int n = n_a;
        const int block_size = std::stoi(argv[4]);
        const std::string mode = argv[5];
        if (block_size <= 0 || block_size > 32) {
            throw std::runtime_error("Block size must be in range 1..32");
        }
        if (mode != "naive" && mode != "tiled") {
            throw std::runtime_error("Mode must be naive or tiled");
        }

        const std::size_t bytes = static_cast<std::size_t>(n) * n * sizeof(double);
        double* device_a = nullptr;
        double* device_b = nullptr;
        double* device_c = nullptr;

        CUDA_CHECK(cudaMalloc(&device_a, bytes));
        CUDA_CHECK(cudaMalloc(&device_b, bytes));
        CUDA_CHECK(cudaMalloc(&device_c, bytes));
        CUDA_CHECK(cudaMemcpy(device_a, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_b, b.data(), bytes, cudaMemcpyHostToDevice));

        const dim3 block(block_size, block_size);
        const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

        cudaEvent_t started;
        cudaEvent_t finished;
        CUDA_CHECK(cudaEventCreate(&started));
        CUDA_CHECK(cudaEventCreate(&finished));
        CUDA_CHECK(cudaEventRecord(started));

        if (mode == "naive") {
            matmul_naive<<<grid, block>>>(device_a, device_b, device_c, n);
        } else {
            const std::size_t shared_bytes = 2 * block_size * block_size * sizeof(double);
            matmul_tiled<<<grid, block, shared_bytes>>>(device_a, device_b, device_c, n);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(finished));
        CUDA_CHECK(cudaEventSynchronize(finished));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, started, finished));

        Matrix c(static_cast<std::size_t>(n) * n);
        CUDA_CHECK(cudaMemcpy(c.data(), device_c, bytes, cudaMemcpyDeviceToHost));
        write_matrix(argv[3], c, n);

        CUDA_CHECK(cudaEventDestroy(started));
        CUDA_CHECK(cudaEventDestroy(finished));
        CUDA_CHECK(cudaFree(device_a));
        CUDA_CHECK(cudaFree(device_b));
        CUDA_CHECK(cudaFree(device_c));

        std::cout << "size=" << n << '\n';
        std::cout << "block_size=" << block_size << '\n';
        std::cout << "mode=" << mode << '\n';
        std::cout << "time_ms=" << std::fixed << std::setprecision(3) << elapsed_ms << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
