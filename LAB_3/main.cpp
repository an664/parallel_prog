#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
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

void multiply_rows(const Matrix& local_a, const Matrix& b_transposed, Matrix& local_c, int rows, int n) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += local_a[static_cast<std::size_t>(i) * n + k] *
                       b_transposed[static_cast<std::size_t>(j) * n + k];
            }
            local_c[static_cast<std::size_t>(i) * n + j] = sum;
        }
    }
}

std::vector<int> make_row_counts(int n, int processes) {
    std::vector<int> rows(processes, n / processes);
    for (int rank = 0; rank < n % processes; ++rank) {
        ++rows[rank];
    }
    return rows;
}

std::vector<int> make_displacements(const std::vector<int>& counts) {
    std::vector<int> displacements(counts.size(), 0);
    for (std::size_t i = 1; i < counts.size(); ++i) {
        displacements[i] = displacements[i - 1] + counts[i - 1];
    }
    return displacements;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int processes = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    if (argc != 4) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <matrix_a.txt> <matrix_b.txt> <result.txt>\n";
        }
        MPI_Finalize();
        return 1;
    }

    int n = 0;
    int status = 0;
    Matrix a;
    Matrix b;

    if (rank == 0) {
        try {
            int n_a = 0;
            int n_b = 0;
            a = read_matrix(argv[1], n_a);
            b = read_matrix(argv[2], n_b);
            if (n_a != n_b) {
                throw std::runtime_error("Matrices must have the same size");
            }
            n = n_a;
        } catch (const std::exception& error) {
            std::cerr << "Error: " << error.what() << '\n';
            status = 1;
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status != 0) {
        MPI_Finalize();
        return 1;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        b.resize(static_cast<std::size_t>(n) * n);
    }
    MPI_Bcast(b.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const std::vector<int> row_counts = make_row_counts(n, processes);
    const std::vector<int> row_displacements = make_displacements(row_counts);
    std::vector<int> send_counts(processes);
    std::vector<int> send_displacements(processes);
    for (int i = 0; i < processes; ++i) {
        send_counts[i] = row_counts[i] * n;
        send_displacements[i] = row_displacements[i] * n;
    }

    const int local_rows = row_counts[rank];
    Matrix local_a(static_cast<std::size_t>(local_rows) * n);
    Matrix local_c(static_cast<std::size_t>(local_rows) * n, 0.0);

    MPI_Scatterv(
        rank == 0 ? a.data() : nullptr,
        send_counts.data(),
        send_displacements.data(),
        MPI_DOUBLE,
        local_a.data(),
        local_rows * n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    const Matrix b_transposed = transpose(b, n);
    MPI_Barrier(MPI_COMM_WORLD);
    const double started = MPI_Wtime();
    multiply_rows(local_a, b_transposed, local_c, local_rows, n);
    MPI_Barrier(MPI_COMM_WORLD);
    const double local_elapsed = MPI_Wtime() - started;

    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    Matrix c;
    if (rank == 0) {
        c.resize(static_cast<std::size_t>(n) * n);
    }
    MPI_Gatherv(
        local_c.data(),
        local_rows * n,
        MPI_DOUBLE,
        rank == 0 ? c.data() : nullptr,
        send_counts.data(),
        send_displacements.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    if (rank == 0) {
        try {
            write_matrix(argv[3], c, n);
            const double operations = 2.0 * std::pow(static_cast<double>(n), 3.0);
            std::cout << "size=" << n << '\n';
            std::cout << "processes=" << processes << '\n';
            std::cout << "operations=" << std::fixed << std::setprecision(0) << operations << '\n';
            std::cout << "time_sec=" << std::setprecision(6) << elapsed << '\n';
        } catch (const std::exception& error) {
            std::cerr << "Error: " << error.what() << '\n';
            MPI_Finalize();
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
