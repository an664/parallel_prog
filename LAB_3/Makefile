MPICXX ?= mpicxx
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra

all: matrix_mpi

matrix_mpi: main.cpp
	$(MPICXX) $(CXXFLAGS) main.cpp -o matrix_mpi

clean:
	rm -f matrix_mpi
