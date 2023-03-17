#pragma once
#ifndef RADIX_SORT_TOOL_H
#define RADIX_SORT_TOOL_H
#include <iostream>

class RadixSortTool {
public:
	RadixSortTool();
	RadixSortTool(int num_bins, int num_bits, int num_base);
	
	~RadixSortTool();

	void SetNumBins(int num_bins);
	void SetNumBits(int num_bits);
	void SetNumBase(int num_base);

	int GetNumBins();
	int GetNumBits();
	int GetNumBase();

	void Sort(int *input_array, int size); // radix sort for input_array, size repr. size of input_array

private:
	int numBins; // max bits
	int numBits; // each iter view bits 
	int numBase; // base number
	int interval; // numBase^numBits

	int* binHistogram; // binHistogram array record number of each num
	int* binScan; // binSacn array record offset
};
#endif