#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

#define DATA 0
#define NUMBER_OF_CHANNELS 1
#define SIZE_OF_BUFFER 8

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
		double complex inputNumbers[NUMBER_OF_CHANNELS][SIZE_OF_BUFFER];
		double complex results[NUMBER_OF_CHANNELS][SIZE_OF_BUFFER];

		FILE* stream = fopen("input.txt", "r");

		//load data from input file
		for (int i = 0; i < SIZE_OF_BUFFER; i++) {
			for (int j = 0; j < NUMBER_OF_CHANNELS; j++) {
				int input;
				int dummy = fscanf(stream, "%d", &input);
				inputNumbers[j][i] = input + 0.0 * I;
			}
		}
		
		//print out input data
		printf("input data\n");
		for (int i = 0; i < NUMBER_OF_CHANNELS; i++) {
			printf("channel %d: ", i + 1);
			for (int j = 0; j < SIZE_OF_BUFFER; j++) {
				printf("%.2f %.2fi", creal(inputNumbers[i][j]), cimag(inputNumbers[i][j]));
				if (j != SIZE_OF_BUFFER - 1) {
					printf(", ");
				}
				else {
					printf("\n");
				}
			}
		}

		//send data to workers
		for (int i = 1; i < NUMBER_OF_CHANNELS + 1; i++) {
			printf("sending data to worker %d\n", i);
			MPI_Send(&inputNumbers[i-1], SIZE_OF_BUFFER, MPI_DOUBLE_COMPLEX, i, DATA, MPI_COMM_WORLD);
		}
		fclose(stream);
		
		//gather results from workers
		for (int i = 1; i < NUMBER_OF_CHANNELS + 1; i++) {
			printf("waiting for data from worker %d\n", i);
			MPI_Recv(&results[i-1], SIZE_OF_BUFFER, MPI_DOUBLE_COMPLEX, i, DATA, MPI_COMM_WORLD, &status);
		}
		printf("received data from all workers\n");
		
		//print out results
		for (int i = 0; i < NUMBER_OF_CHANNELS; i++) {
			printf("worker %d results: ", i + 1);
			for (int j = 0; j < SIZE_OF_BUFFER; j++) {
				printf("%.2f %.2fi", creal(results[i][j]), cimag(results[i][j]));
				if (j != SIZE_OF_BUFFER - 1) {
					printf(", ");
				}
				else {
					printf("\n");
				}
			}
		}
	} else {//worker
		double complex inputNumbers[SIZE_OF_BUFFER];
		//printf("waiting for data, worker %d\n", myRank);
		MPI_Recv(&inputNumbers, SIZE_OF_BUFFER, MPI_DOUBLE_COMPLEX, MPI_ANY_SOURCE, DATA, MPI_COMM_WORLD, &status);
		printf("received data, worker %d, data: \n", myRank);
		
		//print out input data
		int i, j;
		
		for (int i = 0; i < SIZE_OF_BUFFER; i++) {
			printf("%.2f %.2fi", creal(inputNumbers[i]), cimag(inputNumbers[i]));
			if (i != SIZE_OF_BUFFER - 1) {
				printf(", ");
			}
			else {
				printf("\n");
			}
		}
		
		//kernel invocation...
		double complex outputNumbers[SIZE_OF_BUFFER];

		for (i = 0; i < SIZE_OF_BUFFER; i++) {
			outputNumbers[i] = 0;

			for (j = 0; j < SIZE_OF_BUFFER; j++) {
				outputNumbers[i] += inputNumbers[j] * cexp(-2 * M_PI * I * i * j / SIZE_OF_BUFFER);
			}
		}
		
		MPI_Send(&outputNumbers, SIZE_OF_BUFFER, MPI_DOUBLE_COMPLEX, 0, DATA, MPI_COMM_WORLD);	
	}

	MPI_Finalize();
	return 0;
}

