#!/bin/bash
echo "running...";

NP=$(($2+1))
/opt/mpich/ch-p4/bin/mpirun -np $NP ./main $1 $2

