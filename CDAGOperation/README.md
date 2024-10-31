`sudo apt install libgmp-dev`
`g++ -DMAX_NODES=20 -DMAX_CHILDREN=5 -DLOG_TIMINGS=0 -shared -o CDAGOperation.so -fPIC CDAGOperation.cpp  -lgmp`