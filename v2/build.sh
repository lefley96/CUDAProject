#!/bin/bash
echo "building...";

nvcc -arch=sm_20 -c dft.cu -o dft.o
mpicc -c main.c -o main.o -lm
mpicc main.o dft.o -o main -lcudart
rm dft.o main.o

