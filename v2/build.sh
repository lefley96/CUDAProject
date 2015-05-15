#!/bin/bash
echo "building...";

nvcc -arch=sm_20 -c dft.cu -o dft.o
/opt/mpich/ch-p4/bin/mpicc -c main.c -o main.o -lm
/opt/mpich/ch-p4/bin/mpicc main.o dft.o -o main -L/usr/local/cuda/targets/x86_64-linux/lib -L/usr/lib64/ -lcudart -lstdc++
rm dft.o main.o

