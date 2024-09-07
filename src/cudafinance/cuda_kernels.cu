#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        if((call) != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void computeSMA(const float *input, float *output, int numElements, int windowSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        float sum = 0.0;
        int count = 0;
        for (int j = max(0, i - windowSize + 1); j <= i; ++j) {
            sum += input[j];
            count++;
        }
        output[i] = sum / count;
    }
}

void launchSMA_CUDA(const float* h_input, float* h_output, int numElements, int windowSize) {
    float *d_input = NULL, *d_output = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_input, numElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_output, numElements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, numElements * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    computeSMA<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, numElements, windowSize);

    CUDA_CHECK(cudaMemcpy(h_output, d_output, numElements * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}
