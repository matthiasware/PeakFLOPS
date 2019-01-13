all: test_flops peak_flops

GCCFLAGS=-O3 -std=c++11 -march=native

test_flops: test_flops.cc
	g++ -o $@ $^  $(GCCFLAGS)

peak_flops: peak_flops.cc
	g++ -o $@ $^  $(GCCFLAGS)