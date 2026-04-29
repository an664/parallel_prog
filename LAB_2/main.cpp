#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

using Matrix = std::vector<double>;

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

Matrix transpose(const Matrix& matrix, int n) {
    Matrix transposed(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            transposed[static_cast<std::size_t>(j) * n + i] =
                matrix[static_cast<std::size_t>(i) * n + j];
        }
    }
    return transposed;
}

Matrix multiply_openmp(const Matrix& a, const Matrix& b_transposed, int n) {
    Matrix c(static_cast<std::size_t>(n) * n, 0.0);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += a[static_cast<std::size_t>(i) * n + k] *
                       b_transposed[static_cast<std::size_t>(j) * n + k];
            }
            c[static_cast<std::size_t>(i) * n + j] = sum;
        }
    }

    return c;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix_a.txt> <matrix_b.txt> <result.txt> <threads>\n";
        return 1;
    }

    try {
        const int threads = std::stoi(argv[4]);
        if (threads <= 0) {
            throw std::runtime_error("Thread count must be positive");
        }
        omp_set_num_threads(threads);

        int n_a = 0;
        int n_b = 0;
        const Matrix a = read_matrix(argv[1], n_a);
        const Matrix b = read_matrix(argv[2], n_b);
        if (n_a != n_b) {
            throw std::runtime_error("Matrices must have the same size");
        }

        const Matrix b_transposed = transpose(b, n_a);
        const double started = omp_get_wtime();
        const Matrix c = multiply_openmp(a, b_transposed, n_a);
        const double elapsed = omp_get_wtime() - started;

        write_matrix(argv[3], c, n_a);

        const double operations = 2.0 * std::pow(static_cast<double>(n_a), 3.0);
        std::cout << "size=" << n_a << '\n';
        std::cout << "threads=" << threads << '\n';
        std::cout << "operations=" << std::fixed << std::setprecision(0) << operations << '\n';
        std::cout << "time_sec=" << std::setprecision(6) << elapsed << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
