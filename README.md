# Параллельное программирование

## Сведения о студенте

- Студент: Мась Андрей Алексеевич
- Группа: 6311
- Зачетная книжка: 2023-01326

Репозиторий содержит лабораторные работы по курсу «Параллельное программирование».
Основная вычислительная задача во всех работах: перемножение двух квадратных матриц
без использования библиотечной функции умножения в основной C/C++/OpenCL реализации.

Всего в курсе 5 лабораторных работ. Пока в репозитории загружены первые 4
лабораторные работы, а для лабораторной работы №5 добавлена заготовка.

## Структура

| Папка | Тема | Отчет |
|---|---|---|
| `LAB_1` | Последовательное умножение матриц | [LAB_1/README.md](./LAB_1/README.md) |
| `LAB_2` | OpenMP-версия | [LAB_2/README.md](./LAB_2/README.md) |
| `LAB_3` | MPI-версия | [LAB_3/README.md](./LAB_3/README.md) |
| `LAB_4` | OpenCL-версия для GPU | [LAB_4/README.md](./LAB_4/README.md) |
| `LAB_5` | MPI на суперкомпьютере | [LAB_5/README.md](./LAB_5/README.md) (в процессе) |

## Сводное сравнение

Для сравнения методов используется размер `N=2000`, так как на нем влияние
накладных расходов меньше, а вычислительная часть выражена сильнее.

| Метод | Время на N=2000 | Производительность на N=2000 | Ускорение к sequential |
|---|---:|---:|---:|
| Sequential | 7.282340 с | 2.20 GFLOPS | 1.0x |
| Best OpenMP | 1.010286 с | 15.84 GFLOPS | 7.2x |
| Best MPI | 1.138494 с | 14.05 GFLOPS | 6.4x |
| Best OpenCL | 0.060568 с | 264.17 GFLOPS | 120.2x |

Полные графики сравнения CPU/GPU приведены в отчете по [лабораторной работе №4](./LAB_4/README.md).

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
эталонный результат через NumPy и сравнивает его с результатом C/C++/OpenCL
программы через `numpy.allclose`.

Каждая экспериментальная конфигурация запускается 3 раза. В таблицах и на
графиках основным временем считается медиана повторов, а в CSV дополнительно
сохраняются среднее, минимум и стандартное отклонение.

## Среда локальных экспериментов

- ОС: macOS 26.4.1.
- Доступное число логических CPU из Python: 10.
- C++: `c++` / Apple clang для последовательной версии.
- OpenMP: Homebrew GCC `g++-15`.
- MPI: Open MPI 5.0.8 через `mpicxx` / `mpirun`.
- Python: NumPy, pandas, matplotlib.
- GPU: Apple M1 Pro, 16 GPU cores.
- OpenCL: системный Apple OpenCL framework. CUDA локально невозможна, потому что
  CUDA поддерживается только NVIDIA GPU.

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

Для OpenCL:

```bash
cd LAB_4
make
./matrix_opencl sample_A.txt sample_B.txt result.txt 16 tiled
python3 verify.py sample_A.txt sample_B.txt result.txt
```
