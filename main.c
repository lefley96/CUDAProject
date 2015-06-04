#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>

#define DATA 0

void dft(float* samples, int size, float* real, float* imag);

int main(int argc, char **argv) {
    int myRank, procCount;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &procCount);

    int i, j;

    int samples;
    int channels;

    float** output_real;
    float** output_imag;

    // master reads data from the input file and splits data among workers
	if (myRank == 0) {
		if (argc < 3) {
			printf("please specify number of samples and channels as arguments\n");
			return 0;
		}

		samples = atoi(argv[1]);
		channels = atoi(argv[2]);

		printf("Reading file with %d channels of %d samples each\n", channels, samples);

		float** input = (float**) malloc(channels * sizeof(float*));

        for (i = 0; i < channels; i++) {
			input[i] = (float*) malloc(samples * sizeof(float));
        }

		FILE* stream = fopen("input.txt", "r");

		// load data from input file
		for (i = 0; i < samples; i++) {
			for (j = 0; j < channels; j++) {
				int dummy = fscanf(stream, "%f", &input[j][i]);
			}
		}

        fclose(stream);

        // print channels and samples
		for (i = 0; i < channels; i++) {
			printf("Channel %d: ", i + 1);
			
			for (j = 0; j < samples; j++) {
				printf("%.0f", input[i][j]);
						
				if (j != samples - 1) {
						printf(", ");
				}
				else {
					printf("\n");
				}
			}
		}

        printf("\n");
        
		// send data to workers
		for (i = 0; i < channels; i++) {
			MPI_Send(&samples, 1, MPI_INT, i + 1, DATA, MPI_COMM_WORLD);
			MPI_Send(input[i], samples, MPI_FLOAT, i + 1, DATA, MPI_COMM_WORLD);
		}

		// allocate memory for computation results
        output_real = (float**) malloc(channels * sizeof(float*));
        output_imag = (float**) malloc(channels * sizeof(float*));
        for (i = 0; i < channels; i++) {
            output_real[i] = (float*) malloc(samples * sizeof(float));
            output_imag[i] = (float*) malloc(samples * sizeof(float));
        }

        // gather results from workers
		for (i = 0; i < channels; i++) {
			MPI_Recv(output_real[i], samples, MPI_FLOAT, i + 1, DATA, MPI_COMM_WORLD, &status);
			MPI_Recv(output_imag[i], samples, MPI_FLOAT, i + 1, DATA, MPI_COMM_WORLD, &status);
		}
	}
	
    // worker computes data from each channel
    else {
        int received_samples;
        MPI_Recv(&received_samples, 1, MPI_INT, MPI_ANY_SOURCE, DATA, MPI_COMM_WORLD, &status);

        float* received_input = (float*) malloc(received_samples * sizeof(float));
        MPI_Recv(received_input, received_samples, MPI_FLOAT, MPI_ANY_SOURCE, DATA, MPI_COMM_WORLD, &status);

		// gpu kernel invocation
        float* output_real = (float*) malloc(received_samples * sizeof(float));
        float* output_imag = (float*) malloc(received_samples * sizeof(float));
        dft(received_input, received_samples, output_real, output_imag);

        MPI_Send(output_real, received_samples, MPI_FLOAT, 0, DATA, MPI_COMM_WORLD);
        MPI_Send(output_imag, received_samples, MPI_FLOAT, 0, DATA, MPI_COMM_WORLD);
        
        free(received_input);
        free(output_real);
        free(output_imag);
	}

    MPI_Barrier(MPI_COMM_WORLD);

    if (myRank == 0) {
		printf("Received results from all workers\n");

		// print out results
        for (i = 0; i < channels; i++) {
			printf("Channel %d: ", i + 1);
			
			for (j = 0; j < samples; j++) {
				printf("%.2f %.2fi", output_real[i][j], output_imag[i][j]);

				if (j != samples - 1) {
					printf(", ");
				}
				else {
					printf("\n");
				}
			}
		}
    }

	MPI_Finalize();
	return 0;
}
