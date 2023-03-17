#include <iostream>
#include "RadixSortTool.hpp"

int main()
{
    RadixSortTool radix_sort_tool; // instantiation

    // test1
    int input_array1[] = { 981, 213, 3124, 124, 124, 656 };
    std::cout << "Test1" << std::endl;
    std::cout << "Before Radix Sort: ";
    for (int i = 0; i < 6; ++i) {
        std::cout << input_array1[i] << " ";
    }
    std::cout << std::endl;

    radix_sort_tool.Sort(input_array1, 6);
    std::cout << "After Radix Sort: ";
    for (int i = 0; i < 6; ++i) {
        std::cout << input_array1[i] << " ";
    }
    std::cout << std::endl << std::endl;

    // test2 
    int input_array2[] = { 1412, 12412, 1428658, 68, 8658, 56798, 568765, 679234, 355 };
    std::cout << "Test2" << std::endl;
    std::cout << "Before Radix Sort: ";
    for (int i = 0; i < 9; ++i) {
        std::cout << input_array2[i] << " ";
    }
    std::cout << std::endl;

    radix_sort_tool.SetNumBase(2);
    radix_sort_tool.SetNumBins(30);
    radix_sort_tool.SetNumBits(2);

    radix_sort_tool.Sort(input_array2, 9);
    std::cout << "After Radix Sort: ";
    for (int i = 0; i < 9; ++i) {
        std::cout << input_array2[i] << " ";
    }
    std::cout << std::endl << std::endl;

    system("pause");
    return 0;
}