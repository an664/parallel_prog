CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
MPICXX ?= mpicxx

TARGET = matrix_mpi_cluster

.PHONY: all clean

all: $(TARGET)

$(TARGET): main.cpp
	$(MPICXX) $(CXXFLAGS) main.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)
