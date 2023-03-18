# C++ single thread radix sort

**At present, the demo witch will get ascending or descending ordered result through LSD Radix Sort Method is only suit for Integer array.**

You can instantiate class RadixSortTool, set number of base, number of bins(i.e. length of number in base), number of bits for one iter(i.e. added number in each iteration) by Set method.

You also can get number above throgh Get method.

When you finally set these number above, you can get sorted array through Sort method with aguments input_array(**integer array**) and size(i.e. the length of input_array).

In cpu_single_RadixSort.cpp, i give two test samples.

Test1
- numBins: 9
- numBits: 1
- numBase: 10
- sortType: ascending
- sortMethod: LSD
- startPos: 0
- endPos: 6
- input: {981, 213, 3124, 124, 124, 656}
- output: {124, 124, 213, 656, 981, 3124}

Test2
- numBins: 30
- numBits: 2
- numBase: 2
- sortType: ascending
- sortMethod: LSD
- startPos: 0
- endPos: 9
- input: {1412, 12412, 1428658, 68, 8658, 56798, 568765, 679234, 355}
- output: {68, 355, 1412, 8658, 12412, 56798, 568765, 679234, 1428658}

Test3
- numBins: 30
- numBits: 2
- numBase: 2
- sortType: descending
- sortMethod: MSD
- startPos: 2
- endPos: 5
- input: {124 325 235 436 363 3767 898 15 85}
- output: {124 325 436 363 235 3767 898 15 85}
