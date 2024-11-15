# Compile shared library

```bash
sudo apt install libgmp-dev
g++ -DMAX_NODES=20 -DMAX_CHILDREN=5 -DLOG_TIMINGS=0 -DVERBOSE=0 -shared -o libCDAGOperation.so -fPIC -O3 CDAGOperation.cpp  -lgmp
```

# Installing and using C/C++ memory profiling tools

## Installation of Valgrind
```bash
sudo apt install valgrind -y
sudo add-apt-repository ppa:kubuntu-ppa/backports
sudo apt-get update && sudo apt-get install massif-visualizer -y
```

## Usage

Compile the shared library and profiling program with symbols
```bash
g++ -DMAX_NODES=20 -DMAX_CHILDREN=5 -DLOG_TIMINGS=0 -shared -o libCDAGOperation.debug.so -fPIC -g CDAGOperation.cpp  -lgmp
g++ -g -o profiling profiling.cpp -L. -lCDAGOperation.debug -Wl,-rpath=.
```

Run Valgring
```bash
valgrind --tool=massif --time-unit=ms --massif-out-file=massif.out ./profiling
```

Visualize results (graphically or by text)
```bash
massif-visualizer massif.out
ms_print massif.out
```