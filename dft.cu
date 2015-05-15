#include <stdio.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
 
#define THREADS_PER_BLOCK 512

__constant__ float EXPCONST = (-2.0 * 3.141592653);
 
__device__ __forceinline__ cuComplex my_cexpf(cuComplex z) {
    cuComplex res;
    float t = expf(z.x);
 
    sincosf(z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
	
    return res;
}
 
__global__ void dftKernel(cuComplex *input, cuComplex *output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size) {
        cuComplex tempSum = make_cuComplex(0.0, 0.0);
        for (int i = 0; i < size; i++) {
            tempSum = cuCaddf(tempSum, cuCmulf(input[i], my_cexpf(make_cuComplex(0.0, (EXPCONST * i * idx / size)))));
        }
        output[idx] = tempSum;
    }
}
 
extern "C" void dft(float* samples, int size, float* real, float* imag) {
	cuComplex* complex_samples = (cuComplex*) malloc(size * sizeof(cuComplex));

	int i;
	for (i = 0; i < size; i++) {
		complex_samples[i] = make_cuComplex(samples[i], 0);
	}

	cuComplex *d_input, *d_output;
 
	cudaMalloc((void **) &d_input, size * sizeof(cuComplex));
	cudaMalloc((void **) &d_output, size * sizeof(cuComplex));

	cudaMemcpy(d_input, complex_samples, size * sizeof(cuComplex), cudaMemcpyHostToDevice);

	dftKernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_input, d_output, size);

	cudaDeviceSynchronize();

	cudaMemcpy(complex_samples, d_output, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);

	for (i = 0; i < size; i++) {
		real[i] = cuCrealf(complex_samples[i]);
		imag[i] = cuCimagf(complex_samples[i]);
	}

	cudaFree(d_input);
	cudaFree(d_output);
}

/*int main() {
	float a[8] = {92, 79, 68, 32, 16, 40, 7, 87};
	float r[8];
	float i[8];
	
	dft(a, 8, r, i);
 
	printf("output\n");

	int j;
	for (j = 0; j < 8; j++) {
        	printf("%d: %f %f %f\n", j + 1, a[j], r[j], i[j]);
	}

	return 0;
}*/

