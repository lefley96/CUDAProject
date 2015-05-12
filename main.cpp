#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DATA 0
#define NUMBER_OF_CHANNELS 4
#define SIZE_OF_BUFFER 100

int main(int argc, char **argv) {
	int myRank, procCount;
	double range;
	double result = 0.0;

	MPI_Status status;
	// Initialize MPI
	MPI_Init(&argc, &argv);
	// find out my rank
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	// find out the number of processes in MPI_COMM_WORLD
	MPI_Comm_size(MPI_COMM_WORLD, &procCount);

	if (myRank == 0) { //master. reads data and splits into slave threads.
		int inputNumbers[NUMBER_OF_CHANNELS][SIZE_OF_BUFFER];
		FILE* stream = fopen("input.txt", "r");

		for (int i = 0; i < SIZE_OF_BUFFER; i++) {
			for (int j = 0; j < NUMBER_OF_CHANNELS; j++) {
				fscanf(stream, "%d", &inputNumbers[j][i]);
			}
		}

		for (int i = 1; i < NUMBER_OF_CHANNELS + 1; i++) {
			printf("sending data to worker %d\n", i);
			MPI_Send(&inputNumbers[i-1], SIZE_OF_BUFFER, MPI_INT, i, DATA,
					MPI_COMM_WORLD);
		}

	} else {//worker
		int inputNumbers[SIZE_OF_BUFFER];
		printf("waiting for data, worker %d\n", myRank);
		MPI_Recv(&inputNumbers, SIZE_OF_BUFFER, MPI_INT, MPI_ANY_SOURCE, DATA,
				MPI_COMM_WORLD, &status);
		printf("received data, worker %d\n", myRank);
		//kernel invocation...
		printf("worker %d %d %d\n", myRank, inputNumbers[0], inputNumbers[SIZE_OF_BUFFER-1]);		
	}

	MPI_Finalize();
	return 0;
}

