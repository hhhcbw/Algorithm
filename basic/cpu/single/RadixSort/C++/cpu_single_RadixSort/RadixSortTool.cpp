#include <iostream>
#include <assert.h>
#include "RadixSortTool.hpp"

RadixSortTool::RadixSortTool() : numBins(9), numBits(1), numBase(10)
{
	interval = 1;
	for (int i = 0; i < numBits; ++i) {
		interval *= numBase;
	}
	binHistogram = new int[interval];
	binScan = new int[interval];
}

RadixSortTool::RadixSortTool(int num_bins, int num_bits, int num_base)
	: numBins(num_bins), numBits(num_bits), numBase(num_base)
{
	interval = 1;
	for (int i = 0; i < numBits; ++i) {
		interval *= numBase;
	}
	binHistogram = new int[interval];
	binScan = new int[interval];
}

void RadixSortTool::SetNumBins(int num_bins)
{
	numBins = num_bins;
}

void RadixSortTool::SetNumBits(int num_bits)
{
	numBits = num_bits;
}

void RadixSortTool::SetNumBase(int num_base)
{
	numBase = num_base;
}

int RadixSortTool::GetNumBins()
{
	return numBins;
}

int RadixSortTool::GetNumBits()
{
	return numBits;
}

int RadixSortTool::GetNumBase()
{
	return numBase;
}

void RadixSortTool::Sort(int *input_array, int start_pos, int end_pos, int sort_type, int sort_method)
{
	assert(numBins % numBits == 0); // numBin must integral times of numBits
	assert(start_pos >= 0);
	assert(end_pos >= start_pos);

	int size = end_pos - start_pos;
	int* auxArray = new int[size]; // auxiliary array
	int baseNum = 1;

	switch (sort_method)
	{
	case LeastSignificantDigital: // LSD
		for (int i = 0; i < numBins; i += numBits) {
			for (int j = 0; j < interval; ++j) {
				binHistogram[j] = binScan[j] = 0;
			}

			for (int j = start_pos; j < end_pos; ++j) { // count number of each num of i-th bit of input_array
				int temp = (input_array[j] / baseNum) % interval; // Get i-th bit of each number in input_array
				binHistogram[temp] ++;
			}

			for (int j = 1; j < interval; ++j) { // calculate offset as basic index of each num
				binScan[j] = binScan[j - 1] + binHistogram[j - 1];
			}

			for (int j = start_pos; j < end_pos; ++j) { // proceed to sort, insert number into corresponding bucket
				int temp = (input_array[j] / baseNum) % interval;
				auxArray[binScan[temp]++] = input_array[j];
			}

			for (int j = 0; j < size; ++j) { // copy from auxArray to input_array
				input_array[j + start_pos] = auxArray[j];
			}
			//std::swap(input_array, auxArray); // Error, swap will cause delete error

			baseNum *= interval; // update baseNum
		}
		break;

	case MostSignificantDigital: // MSD wating to program
		for (int i = 0; i < numBins; i += numBits) {
			for (int j = 0; j < interval; ++j) {
				binHistogram[j] = binScan[j] = 0;
			}

			for (int j = start_pos; j < end_pos; ++j) { // count number of each num of i-th bit of input_array
				int temp = (input_array[j] / baseNum) % interval; // Get i-th bit of each number in input_array
				binHistogram[temp] ++;
			}

			for (int j = 1; j < interval; ++j) { // calculate offset as basic index of each num
				binScan[j] = binScan[j - 1] + binHistogram[j - 1];
			}

			for (int j = start_pos; j < end_pos; ++j) { // proceed to sort, insert number into corresponding bucket
				int temp = (input_array[j] / baseNum) % interval;
				auxArray[binScan[temp]++] = input_array[j];
			}

			for (int j = 0; j < size; ++j) { // copy from auxArray to input_array
				input_array[j + start_pos] = auxArray[j];
			}
			//std::swap(input_array, auxArray); // Error, swap will cause delete error

			baseNum *= interval; // update baseNum
		}
		break;

	default:
		break;
	}

	if (sort_type == DESCENDING) { // if sort_type is descending reverse input_array
		for (int i = 0; i < size; ++i) {
			input_array[i + start_pos] = auxArray[size - i - 1];
		}
	}

	delete [] auxArray;
}

RadixSortTool::~RadixSortTool() {
	delete [] binHistogram;
	delete [] binScan;
}