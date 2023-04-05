#include <iostream>
#include <atomic>
#include <thread>
#include <vector>
#include <windows.h>

#define SIZE 100000000
#define numBins 8
#define numBits 1
#define numBase 100
#define THREADS 10

std::atomic<int> binHistogram[THREADS][numBase];
std::atomic<int> binScan[THREADS][numBase];

void ClearBin() {
    for (int i=0; i<THREADS; ++i)
        for (int j = 0; j < numBase; ++j) {
            binHistogram[i][j].store(0);
            binScan[i][j].store(0);
        }
}

void CalbinHistogram(int* _inputVals, int _thread_idx, int _block_start, int _block_end, int _baseNum)
{
    for (int i = _block_start; i < _block_end; ++i) {
        int temp = (_inputVals[i] / _baseNum) % numBase;
        binHistogram[_thread_idx][temp] ++;
    }
}

void CaloutputVals(int* _inputVals, int* _outputVals, int _thread_idx, int _block_start, int _block_end, int _baseNum)
{
    for (int i = _block_start; i < _block_end; ++i) {
        int temp = (_inputVals[i] / _baseNum) % numBase;
        _outputVals[binScan[_thread_idx][temp]++] = _inputVals[i];
    }
}

void SwapinputVals(int* _inputVals, int* _outputVals, int _block_start, int _block_end) {
    for (int i = _block_start; i < _block_end; ++i) {
        _inputVals[i] = _outputVals[i];
    }
}

void sort(int inputVals[], int outputVals[])
{
    const int hardware_threads = std::thread::hardware_concurrency();
    const int num_threads = (std::min)(hardware_threads != 0 ? hardware_threads : 2, THREADS);
    const int block_size = SIZE / num_threads;

    //std::cout << num_threads << " " << block_size << std::endl;

    std::vector<std::thread> threads(num_threads - 1);

    int baseNum = 1;

    for (int i = 0; i < numBins; i += numBits) {
        ClearBin();
        int block_start = 0;
        int block_end = 0;

        for (int j = 0; j < (num_threads - 1); ++j)
        {
            block_end += block_size;
            threads[j] = std::thread(CalbinHistogram, inputVals, j, block_start, block_end, baseNum);
            block_start = block_end;
        }
        CalbinHistogram(inputVals, num_threads - 1, block_start, SIZE, baseNum);
        for (auto& entry : threads)
            entry.join();

        for (int j = 0; j < numBase; ++j)
            for (int k = 0; k < THREADS; ++k) {
                if (k != 0) {
                    binScan[k][j].store(binScan[k - 1][j] + binHistogram[k - 1][j]);
                }
                else if(j != 0){
                    binScan[k][j].store(binScan[num_threads - 1][j - 1] + binHistogram[num_threads - 1][j - 1]);
                }
            }

        block_start = 0;
        block_end = 0;

        for (int j = 0; j < (num_threads - 1); ++j)
        {
            block_end += block_size;
            threads[j] = std::thread(CaloutputVals, inputVals, outputVals, j, block_start, block_end, baseNum);
            block_start = block_end;
        }
        CaloutputVals(inputVals, outputVals, num_threads - 1, block_start, SIZE, baseNum);
        for (auto& entry : threads)
            entry.join();
        
        block_start = 0;
        block_end = 0;

        for (int j = 0; j < (num_threads - 1); ++j)
        {
            block_end += block_size;
            threads[j] = std::thread(SwapinputVals, inputVals, outputVals, block_start, block_end);
            block_start = block_end;
        }
        SwapinputVals(inputVals, outputVals, block_start, SIZE);
        for (auto& entry : threads)
            entry.join();

        baseNum *= numBase;
    }

}

int main()
{
    //int inputVals[SIZE] = { 234, 2435, 57, 25, 687, 988, 12345, 47543, 1351, 457 };
    int* inputVals = new int[SIZE];
    int* outputVals = new int[SIZE];
    
    for (int i = 0; i < SIZE; ++i) {
        inputVals[i] = SIZE - i - 1;
    }
    //std::cout << "Before Sort: ";
    //for (int i = 0; i < SIZE; ++i) {
    //    std::cout << inputVals[i] << " ";
    //}
    //std::cout << std::endl;
    
    DWORD start_time = GetTickCount64();
    sort(inputVals, outputVals);
    DWORD end_time = GetTickCount64();

    for (int i = 0; i < SIZE; ++i) {
        if (inputVals[i] != i) {
            std::cout << "answer is wrong" << std::endl;
            exit(1);
        }
    }
    std::cout << "answer is right" << std::endl;
    std::cout << "Runtime: " << end_time - start_time << "ms" << std::endl;
    //std::cout << "After Sort: ";
    //for (int i = 0; i < SIZE; ++i) {
    //    std::cout << outputVals[i] << " ";
    //}
    //std::cout << std::endl;
    
    delete[] inputVals;
    delete[] outputVals;
}

