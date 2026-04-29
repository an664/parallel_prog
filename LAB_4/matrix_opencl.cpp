#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Matrix = std::vector<float>;

const char* KERNEL_SOURCE = R"CLC(
__kernel void matmul_naive(__global const float* a,
                           __global const float* b,
                           __global float* c,
                           const int n) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    c[row * n + col] = sum;
}

__kernel void matmul_tiled(__global const float* a,
                           __global const float* b,
                           __global float* c,
                           const int n,
                           __local float* tile_a,
                           __local float* tile_b,
                           const int tile_size) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);
    const int local_col = get_local_id(0);
    const int local_row = get_local_id(1);

    float sum = 0.0f;
    for (int offset = 0; offset < n; offset += tile_size) {
        const int a_col = offset + local_col;
        const int b_row = offset + local_row;

        tile_a[local_row * tile_size + local_col] =
            (row < n && a_col < n) ? a[row * n + a_col] : 0.0f;
        tile_b[local_row * tile_size + local_col] =
            (b_row < n && col < n) ? b[b_row * n + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < tile_size; ++k) {
            sum += tile_a[local_row * tile_size + k] * tile_b[k * tile_size + local_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}
)CLC";

void check_cl(cl_int status, const std::string& message) {
    if (status != CL_SUCCESS) {
        throw std::runtime_error(message + " (OpenCL error " + std::to_string(status) + ")");
    }
}

std::string get_device_name(cl_device_id device) {
    std::size_t size = 0;
    check_cl(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size), "Cannot read device name size");
    std::string name(size, '\0');
    check_cl(clGetDeviceInfo(device, CL_DEVICE_NAME, size, name.data(), nullptr), "Cannot read device name");
    while (!name.empty() && name.back() == '\0') {
        name.pop_back();
    }
    return name;
}

cl_device_id choose_device() {
    cl_uint platform_count = 0;
    check_cl(clGetPlatformIDs(0, nullptr, &platform_count), "Cannot query OpenCL platforms");
    if (platform_count == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }

    std::vector<cl_platform_id> platforms(platform_count);
    check_cl(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "Cannot read OpenCL platforms");

    const cl_device_type preferred_types[] = {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL};
    for (cl_device_type type : preferred_types) {
        for (cl_platform_id platform : platforms) {
            cl_uint device_count = 0;
            const cl_int status = clGetDeviceIDs(platform, type, 0, nullptr, &device_count);
            if (status != CL_SUCCESS || device_count == 0) {
                continue;
            }

            std::vector<cl_device_id> devices(device_count);
            check_cl(clGetDeviceIDs(platform, type, device_count, devices.data(), nullptr),
                     "Cannot read OpenCL devices");
            return devices.front();
        }
    }

    throw std::runtime_error("No OpenCL devices found");
}

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
    for (float& value : matrix) {
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

    out << n << '\n' << std::setprecision(8);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << matrix[static_cast<std::size_t>(i) * n + j];
            out << (j + 1 == n ? '\n' : ' ');
        }
    }
}

std::size_t round_up(int value, int factor) {
    return static_cast<std::size_t>(((value + factor - 1) / factor) * factor);
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <matrix_a.txt> <matrix_b.txt> <result.txt> <local_size> <naive|tiled>\n";
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
        const int local_size = std::stoi(argv[4]);
        const std::string mode = argv[5];
        if (local_size <= 0 || local_size > 32) {
            throw std::runtime_error("Local size must be in range 1..32");
        }
        if (mode != "naive" && mode != "tiled") {
            throw std::runtime_error("Mode must be naive or tiled");
        }

        cl_int status = CL_SUCCESS;
        cl_device_id device = choose_device();
        std::size_t max_work_group_size = 0;
        check_cl(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size),
                                 &max_work_group_size, nullptr),
                 "Cannot read max work-group size");
        if (static_cast<std::size_t>(local_size) * local_size > max_work_group_size) {
            throw std::runtime_error("Requested local size exceeds device work-group limit");
        }

        cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        check_cl(status, "Cannot create OpenCL context");

        cl_command_queue queue =
            clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
        check_cl(status, "Cannot create OpenCL command queue");

        const char* sources[] = {KERNEL_SOURCE};
        const std::size_t source_sizes[] = {std::char_traits<char>::length(KERNEL_SOURCE)};
        cl_program program = clCreateProgramWithSource(context, 1, sources, source_sizes, &status);
        check_cl(status, "Cannot create OpenCL program");

        status = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        if (status != CL_SUCCESS) {
            std::size_t log_size = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            throw std::runtime_error("Cannot build OpenCL program:\n" + log);
        }

        cl_kernel kernel =
            clCreateKernel(program, mode == "naive" ? "matmul_naive" : "matmul_tiled", &status);
        check_cl(status, "Cannot create OpenCL kernel");

        const std::size_t bytes = static_cast<std::size_t>(n) * n * sizeof(float);
        cl_mem device_a =
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                           const_cast<float*>(a.data()), &status);
        check_cl(status, "Cannot create buffer A");
        cl_mem device_b =
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes,
                           const_cast<float*>(b.data()), &status);
        check_cl(status, "Cannot create buffer B");
        cl_mem device_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &status);
        check_cl(status, "Cannot create buffer C");

        check_cl(clSetKernelArg(kernel, 0, sizeof(device_a), &device_a), "Cannot set kernel arg 0");
        check_cl(clSetKernelArg(kernel, 1, sizeof(device_b), &device_b), "Cannot set kernel arg 1");
        check_cl(clSetKernelArg(kernel, 2, sizeof(device_c), &device_c), "Cannot set kernel arg 2");
        check_cl(clSetKernelArg(kernel, 3, sizeof(n), &n), "Cannot set kernel arg 3");
        if (mode == "tiled") {
            const std::size_t tile_bytes = static_cast<std::size_t>(local_size) * local_size * sizeof(float);
            check_cl(clSetKernelArg(kernel, 4, tile_bytes, nullptr), "Cannot set kernel arg 4");
            check_cl(clSetKernelArg(kernel, 5, tile_bytes, nullptr), "Cannot set kernel arg 5");
            check_cl(clSetKernelArg(kernel, 6, sizeof(local_size), &local_size), "Cannot set kernel arg 6");
        }

        const std::size_t global_size[] = {round_up(n, local_size), round_up(n, local_size)};
        const std::size_t local_work_size[] = {
            static_cast<std::size_t>(local_size),
            static_cast<std::size_t>(local_size),
        };

        cl_event event = nullptr;
        check_cl(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, local_work_size, 0, nullptr,
                                        &event),
                 "Cannot enqueue OpenCL kernel");
        check_cl(clFinish(queue), "Cannot finish OpenCL queue");

        cl_ulong started = 0;
        cl_ulong finished = 0;
        check_cl(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(started), &started,
                                         nullptr),
                 "Cannot read kernel start time");
        check_cl(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(finished), &finished,
                                         nullptr),
                 "Cannot read kernel end time");

        Matrix c(static_cast<std::size_t>(n) * n);
        check_cl(clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, bytes, c.data(), 0, nullptr, nullptr),
                 "Cannot read result matrix");
        write_matrix(argv[3], c, n);

        clReleaseEvent(event);
        clReleaseMemObject(device_a);
        clReleaseMemObject(device_b);
        clReleaseMemObject(device_c);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        const double elapsed_ms = static_cast<double>(finished - started) / 1'000'000.0;
        std::cout << "size=" << n << '\n';
        std::cout << "local_size=" << local_size << '\n';
        std::cout << "mode=" << mode << '\n';
        std::cout << "device=" << get_device_name(device) << '\n';
        std::cout << "time_ms=" << std::fixed << std::setprecision(3) << elapsed_ms << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
