# Параллельное программирование

Репозиторий содержит лабораторные работы по курсу «Параллельное программирование».
Основная вычислительная задача во всех работах: перемножение двух квадратных матриц
без использования библиотечной функции умножения в основной C/C++/CUDA реализации.

Всего в курсе 5 лабораторных работ. В репозитории подготовлены первые 4 работы.
Лабораторная работа №5 не включена, потому что для нее нужен доступ к
суперкомпьютеру «Сергей Королев».

## Структура

| Папка | Тема | Отчет |
|---|---|---|
| `LAB_1` | Последовательное умножение матриц | [LAB_1/README.md](./LAB_1/README.md) |
| `LAB_2` | OpenMP-версия | [LAB_2/README.md](./LAB_2/README.md) |
| `LAB_3` | MPI-версия | [LAB_3/README.md](./LAB_3/README.md) |
| `LAB_4` | CUDA-версия | [LAB_4/README.md](./LAB_4/README.md) |

## Общий формат данных

Входной файл содержит размерность `N`, затем `N` строк по `N` чисел:

```text
4
1 2 3 4
5 6 7 8
2 0 1 3
4 1 0 2
```

Выходной файл имеет такой же формат: первая строка содержит размер результирующей
матрицы, далее идет матрица `C = A * B`.

## Верификация

Для каждой лабораторной работы есть скрипт `verify.py`, который вычисляет
эталонный результат через NumPy и сравнивает его с результатом C/C++/CUDA
программы через `numpy.allclose`.

## Среда локальных экспериментов

- ОС: macOS 26.4.1.
- Доступное число логических CPU из Python: 10.
- C++: `c++` / Apple clang для последовательной версии.
- OpenMP: Homebrew GCC `g++-15`.
- MPI: Open MPI 5.0.8 через `mpicxx` / `mpirun`.
- Python: NumPy, pandas, matplotlib.
- CUDA: локально `nvcc` не установлен, NVIDIA GPU недоступен; код и сценарии
  для CUDA подготовлены, но фактический запуск должен выполняться на CUDA-хосте.

## Быстрый запуск

```bash
cd LAB_1
make
./matrix_seq sample_A.txt sample_B.txt result.txt
python3 verify.py sample_A.txt sample_B.txt result.txt
```

Для OpenMP:

```bash
cd LAB_2
make
./matrix_omp sample_A.txt sample_B.txt result.txt 4
python3 verify.py sample_A.txt sample_B.txt result.txt
```

Для MPI:

```bash
cd LAB_3
make
mpirun --map-by slot --oversubscribe -np 4 ./matrix_mpi sample_A.txt sample_B.txt result.txt
python3 verify.py sample_A.txt sample_B.txt result.txt
```

Для CUDA на машине с NVIDIA CUDA Toolkit:

```bash
cd LAB_4
make
./matrix_cuda sample_A.txt sample_B.txt result.txt 16 tiled
python3 verify.py sample_A.txt sample_B.txt result.txt
```
