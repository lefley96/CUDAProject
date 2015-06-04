#include <cuda_runtime.h>
#include <cuComplex.h>

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

int chooseAndSetBestDevice() {
    int num_devices, device;
    int max_multiprocessors = 0, max_device = 0;
    
    cudaGetDeviceCount(&num_devices);
	
    if (num_devices > 1) {          
		for (device = 0; device < num_devices; device++) {
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, device);
			
			if (max_multiprocessors < properties.multiProcessorCount) {
				max_multiprocessors = properties.multiProcessorCount;
				max_device = device;
			}
		}
    }
	
    cudaSetDevice(max_device);
    return max_device; 
}

int getThreadsPerBlock(int currentDevice) {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, currentDevice);
    return properties.maxThreadsPerBlock;
}
 
extern "C" void dft(float* samples, int size, float* real, float* imag) {
    int currentDevice = chooseAndSetBestDevice();
    int THREADS_PER_BLOCK = getThreadsPerBlock(currentDevice);

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

    cudaFree(d_output);
    cudaFree(d_input);
    free(complex_samples);
}

