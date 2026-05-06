#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Matrix = std::vector<double>;

struct Config {
    std::vector<int> sizes{200, 400, 800, 1200, 1600, 2000};
    int repeats = 3;
    int seed = 2026;
    std::string output = "measured_results.csv";
    bool append = false;
};

std::vector<int> parse_list(const std::string& value) {
    std::vector<int> result;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            result.push_back(std::stoi(item));
        }
    }
    if (result.empty()) {
        throw std::runtime_error("List of sizes must not be empty");
    }
    return result;
}

Config parse_args(int argc, char* argv[]) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--sizes" && i + 1 < argc) {
            config.sizes = parse_list(argv[++i]);
        } else if (arg == "--repeats" && i + 1 < argc) {
            config.repeats = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            config.output = argv[++i];
        } else if (arg == "--append") {
            config.append = true;
        } else if (arg == "--help") {
            throw std::runtime_error(
                "Usage: matrix_mpi_cluster [--sizes 200,400,...] [--repeats 3] "
                "[--seed 2026] [--output measured_results.csv] [--append]");
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + arg);
        }
    }
    if (config.repeats <= 0) {
        throw std::runtime_error("Repeat count must be positive");
    }
    return config;
}

double generated_a(int row, int col, int seed) {
    const long long value = (static_cast<long long>(row) * 17 + static_cast<long long>(col) * 31 + seed) % 101;
    return (static_cast<double>(value) - 50.0) / 50.0;
}

double generated_b(int row, int col, int seed) {
    const long long value = (static_cast<long long>(row) * 43 + static_cast<long long>(col) * 13 + seed * 3) % 103;
    return (static_cast<double>(value) - 51.0) / 51.0;
}

Matrix make_matrix(int n, int seed, bool second) {
    Matrix matrix(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[static_cast<std::size_t>(i) * n + j] =
                second ? generated_b(i, j, seed) : generated_a(i, j, seed);
        }
    }
    return matrix;
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

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 1) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double stddev(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 0.0;
    }
    const double avg = mean(values);
    double square_sum = 0.0;
    for (double value : values) {
        const double diff = value - avg;
        square_sum += diff * diff;
    }
    return std::sqrt(square_sum / static_cast<double>(values.size() - 1));
}

bool file_has_data(const std::string& path) {
    std::ifstream input(path);
    return input.good() && input.peek() != std::ifstream::traits_type::eof();
}

void write_header(std::ofstream& output) {
    output << "size,processes,operations,repeats,time_sec,time_median_sec,time_mean_sec,"
           << "time_min_sec,time_std_sec,gflops,data_source\n";
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int processes = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    Config config;
    try {
        config = parse_args(argc, argv);
    } catch (const std::exception& error) {
        if (rank == 0) {
            std::cerr << error.what() << '\n';
        }
        MPI_Finalize();
        return 1;
    }

    std::ofstream csv;
    int csv_status = 0;
    if (rank == 0) {
        const bool need_header = !config.append || !file_has_data(config.output);
        csv.open(config.output, config.append ? std::ios::app : std::ios::out);
        if (!csv) {
            std::cerr << "Cannot open output file: " << config.output << '\n';
            csv_status = 1;
        } else if (need_header) {
            write_header(csv);
        }
    }
    MPI_Bcast(&csv_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (csv_status != 0) {
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        csv << std::fixed << std::setprecision(6);
    }

    for (int n : config.sizes) {
        Matrix a;
        Matrix b;
        if (rank == 0) {
            a = make_matrix(n, config.seed + n, false);
            b = make_matrix(n, config.seed + n, true);
        } else {
            b.resize(static_cast<std::size_t>(n) * n);
        }

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

        std::vector<double> compute_times;

        for (int repeat = 0; repeat < config.repeats; ++repeat) {
            std::fill(local_c.begin(), local_c.end(), 0.0);

            MPI_Barrier(MPI_COMM_WORLD);

            MPI_Bcast(b.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
            const double compute_started = MPI_Wtime();
            multiply_rows(local_a, b_transposed, local_c, local_rows, n);
            MPI_Barrier(MPI_COMM_WORLD);
            const double local_compute = MPI_Wtime() - compute_started;

            double compute_elapsed = 0.0;
            MPI_Reduce(&local_compute, &compute_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

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
                compute_times.push_back(compute_elapsed);
            }
        }

        if (rank == 0) {
            const double time_median = median(compute_times);
            const double operations = 2.0 * std::pow(static_cast<double>(n), 3.0);
            const double gflops = operations / time_median / 1e9;

            csv << n << ',' << processes << ',' << std::setprecision(0) << operations << std::setprecision(6)
                << ',' << config.repeats
                << ',' << time_median
                << ',' << time_median
                << ',' << mean(compute_times)
                << ',' << *std::min_element(compute_times.begin(), compute_times.end())
                << ',' << stddev(compute_times)
                << ',' << std::setprecision(3) << gflops << std::setprecision(6)
                << ",measured_on_current_mpi_environment\n";

            std::cout << "size=" << n
                      << " processes=" << processes
                      << " repeats=" << config.repeats
                      << " compute_median_sec=" << time_median
                      << '\n';
        }
    }

    MPI_Finalize();
    return 0;
}
