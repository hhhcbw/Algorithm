#pragma once
#ifndef RADIX_SORT_TOOL_H
#define RADIX_SORT_TOOL_H
#include <iostream>

enum SortMethod {
LeastSignificantDigital,
MostSignificantDigital
};

enum SortType {
ASCENDING,
DESCENDING
};

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

	// radix sort for input_array
	// input_array: array which will be sorted.
	// start_pos: start position of array to sort
	// end_pos: end position of array to sort [start_pos, end_pos)
	// sort_type: ascending or descending
	// sort_method: LSD or MSD
	void Sort(int *input_array, int start_pos, int end_pos, int sort_type = ASCENDING, int sort_method = LeastSignificantDigital);

private:
	int numBins; // max bits
	int numBits; // each iter view bits 
	int numBase; // base number
	int interval; // numBase^numBits

	int* binHistogram; // binHistogram array record number of each num
	int* binScan; // binSacn array record offset
};
#endif