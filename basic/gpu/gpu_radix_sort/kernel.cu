#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define SIZE 100000
#define THREADS 1024
#define blockSize ((int)ceil((float)SIZE / THREADS))
#define numBits 1
#define numBins 25
#define numBase 2

__global__ void CalBinHistogram(int* bin_histogram, const int* input_vals, const int base_num)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < SIZE) {
        atomicAdd(&bin_histogram[blockIdx.x + (input_vals[tid] / base_num) % numBase * gridDim.x], 1);
    }
}

__global__ void CalBinScan(int* bin_scan, int* bin_histogram)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    int cur_val = bin_histogram[tid];
    int cur_cdf = bin_histogram[tid];
    for (int i = 1; i < blockDim.x; i <<= 1) {
        if (tid >= i)
            cur_cdf += bin_histogram[tid - i];
        __syncthreads();

        bin_histogram[tid] = cur_cdf;
        __syncthreads();
    }

    bin_scan[tid] = bin_histogram[tid] - cur_val;
}

__global__ void CalScanArray(int* bins, const int* input_vals, const int base_num, const int cur_num)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < SIZE) { 
        if ((input_vals[tid] / base_num) % numBase == cur_num) {
            bins[tid] = 1;
        }
        else {
            bins[tid] = 0;
        }
    }
}

__global__ void CalOffset(int* offset, int* bins) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= SIZE)
        return;

    int cur_val = bins[tid];
    int cur_cdf = bins[tid];
    for (int i = 1; i < blockDim.x; i <<= 1) {
        if (tid >= i + blockDim.x * blockIdx.x)
            cur_cdf += bins[tid - i];
        __syncthreads();

        bins[tid] = cur_cdf;
        __syncthreads();
    }

    if (cur_val != 0)
        offset[tid] += bins[tid] - cur_val;

}

__global__ void CalSortArray(int* output_vals, const int* input_vals, const int* bin_scan, const int* offset, const int base_num)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < SIZE) {
        int index = bin_scan[blockIdx.x + (input_vals[tid] / base_num) % numBase * gridDim.x ] + offset[tid];
        output_vals[index] = input_vals[tid];
    }
}

int main(void)
{
    //int inputVals[SIZE] = {149124, 41283, 14765, 14624, 5523, 1244, 3523, 3252, 12532, 92580}, outputVals[SIZE];
    //int inputVals[SIZE] = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }, outputVals[SIZE];
    int inputVals[SIZE], outputVals[SIZE];
    for (int i = 0; i < SIZE; ++i)
        inputVals[i] = SIZE - i - 1;
    //int binHistogram[numBase * 1000], binScan[numBase * 1000], bins[SIZE], offset[SIZE];
    int dev_baseNum = 1;
    int* dev_inputVals, * dev_outputVals, * dev_binHistogram, * dev_binScan, * dev_bins, * dev_offset;

    // alloc memory in device
    cudaMalloc((void**)&dev_inputVals , SIZE * sizeof(int));
    cudaMalloc((void**)&dev_outputVals, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_binHistogram, numBase * blockSize * sizeof(int));
    cudaMalloc((void**)&dev_binScan, numBase * blockSize * sizeof(int));
    cudaMalloc((void**)&dev_bins, SIZE * sizeof(int));
    cudaMalloc((void**)&dev_offset, SIZE * sizeof(int));

    // copy data from host to device
    cudaMemcpy(dev_inputVals, inputVals, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < numBins; i += numBits) {
        cudaMemset(dev_binHistogram, 0, numBase * blockSize * sizeof(int));
        cudaMemset(dev_binScan, 0, numBase * blockSize * sizeof(int));
        cudaMemset(dev_offset, 0, SIZE * sizeof(int));

        //std::cout << blockSize << std::endl;

        CalBinHistogram <<<blockSize, THREADS>>> (dev_binHistogram, dev_inputVals, dev_baseNum);
        //cudaMemcpy(binHistogram, dev_binHistogram, numBase * blockSize * sizeof(int), cudaMemcpyDeviceToHost);
        //for (int j = 0; j < numBase * blockSize; ++j) {
        //    std::cout << binHistogram[j] << " ";
        //}
        //std::cout << std::endl;

        CalBinScan <<<1, numBase * blockSize>>> (dev_binScan, dev_binHistogram);
        //cudaMemcpy(binScan, dev_binScan, numBase * blockSize * sizeof(int), cudaMemcpyDeviceToHost);
        //for (int j = 0; j < numBase * blockSize; ++j) {
        //    std::cout << binScan[j] << " ";
        //}
        //std::cout << std::endl;

        for (int j = 0; j < numBase; ++j) {
            CalScanArray <<<blockSize, THREADS>>> (dev_bins, dev_inputVals, dev_baseNum, j);
            //cudaMemcpy(bins, dev_bins, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
            //for (int k = 0; k < SIZE; ++k) {
            //    std::cout << bins[k] << " ";
            //}
            //std::cout << std::endl;
            CalOffset <<<blockSize, THREADS>>> (dev_offset, dev_bins);
            //cudaMemcpy(offset, dev_offset, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
            //for (int k = 0; k < SIZE; ++k) {
            //    std::cout << offset[k] << " ";
            //}
            //std::cout << std::endl;
        }

        CalSortArray <<<blockSize , THREADS>>> (dev_outputVals, dev_inputVals, dev_binScan, dev_offset, dev_baseNum);
        //cudaMemcpy(outputVals, dev_outputVals, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        //for (int j = 0; j < SIZE; ++j) {
        //    std::cout << outputVals[j] << " ";
        //}
        //std::cout << std::endl;

        cudaMemcpy(dev_inputVals, dev_outputVals, SIZE * sizeof(int), cudaMemcpyDeviceToDevice);
        dev_baseNum *= numBase;
    }

    // copy data from device to host
    cudaMemcpy(outputVals, dev_outputVals , SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // print answer
    //std::cout << "Before sort: " << std::endl;
    //for (int i = 0; i < SIZE; i++) {
    //    std::cout << inputVals[i] << " ";
    //}
    //std::cout << std::endl;
    //std::cout << "After sort: " << std::endl;
    //for (int i = 0; i < SIZE; i++) {
    //    std::cout << outputVals[i] << " ";
    //}
    //std::cout << std::endl;

    // free memory in device
    cudaFree(dev_inputVals);
    cudaFree(dev_outputVals);
    cudaFree(dev_binHistogram);
    cudaFree(dev_binScan);
    cudaFree(dev_bins);
    cudaFree(dev_offset);

    // check answer
    for (int i = 0; i < SIZE; ++i) {
        if (outputVals[i] != i) {
            std::cout << "answer is wrong" << std::endl;
            exit(1);
        }
    }
    std::cout << "answer is right" << std::endl;
    return 0;
}
