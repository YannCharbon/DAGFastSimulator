# Compile shared library

```bash
sudo apt install libgmp-dev
g++ -DMAX_NODES=20 -DMAX_CHILDREN=5 -DLOG_TIMINGS=0 -DVERBOSE=0 -shared -o libCDAGOperation.so -fPIC -Ofast -flto CDAGOperation.cpp  -lgmp
```

### Instructions to install GMP as non-root

```bash
wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
tar -xf gmp-6.3.0.tar.xz
cd gmp-6.3.0/
./configure --prefix=$HOME/local/gmp
make
make install
make check
```

And add the following to `~/.bashrc`

```bash
# GMP
export PATH=$HOME/local/gmp/bin:$PATH
export LD_LIBRARY_PATH=$HOME/local/gmp/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$HOME/local/gmp/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$HOME/local/gmp/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$HOME/local/gmp/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$HOME/local/gmp/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$HOME/local/gmp/lib:$LIBRARY_PATH
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