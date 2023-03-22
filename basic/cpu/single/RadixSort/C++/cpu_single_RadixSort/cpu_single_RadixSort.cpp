#include <iostream>
#include <windows.h>
#include "RadixSortTool.hpp"
#define SIZE 100000000
int main()
{
    RadixSortTool radix_sort_tool; // instantiation

    // test run time
    int *input_array = new int[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        input_array[i] = SIZE - i - 1;
    }
    radix_sort_tool.SetNumBase(10);
    radix_sort_tool.SetNumBins(8);
    radix_sort_tool.SetNumBits(1);
    DWORD start_time = GetTickCount64();
    radix_sort_tool.Sort(input_array, 0, SIZE);
    DWORD end_time = GetTickCount64();
    for (int i = 0; i < SIZE; ++i) {
        if (input_array[i] != i) {
            std::cout << "answer is wrong" << std::endl;
        }
    }
    delete[] input_array;
    std::cout << "answer is right" << std::endl;
    std::cout << "Runtime: " << end_time - start_time << "ms" << std::endl;

    //// test1
    //int input_array1[] = { 981, 213, 3124, 124, 124, 656 };
    //std::cout << sizeof(input_array1) << std::endl;
    //std::cout << "Test1" << std::endl;
    //std::cout << "Before Radix Sort: ";
    //for (int i = 0; i < 6; ++i) {
    //    std::cout << input_array1[i] << " ";
    //}
    //std::cout << std::endl;

    //radix_sort_tool.Sort(input_array1, 0, 6);
    //std::cout << "After Radix Sort: ";
    //for (int i = 0; i < 6; ++i) {
    //    std::cout << input_array1[i] << " ";
    //}
    //std::cout << std::endl << std::endl;

    //// test2 
    //int input_array2[] = { 1412, 12412, 1428658, 68, 8658, 56798, 568765, 679234, 355 };
    //std::cout << "Test2" << std::endl;
    //std::cout << "Before Radix Sort: ";
    //for (int i = 0; i < 9; ++i) {
    //    std::cout << input_array2[i] << " ";
    //}
    //std::cout << std::endl;

    //radix_sort_tool.SetNumBase(2);
    //radix_sort_tool.SetNumBins(30);
    //radix_sort_tool.SetNumBits(2);

    //radix_sort_tool.Sort(input_array2, 0, 9);
    //std::cout << "After Radix Sort: ";
    //for (int i = 0; i < 9; ++i) {
    //    std::cout << input_array2[i] << " ";
    //}
    //std::cout << std::endl << std::endl;

    //// test3
    //int input_array3[] = { 124, 325, 235, 436, 363, 3767, 898, 15, 85 };
    //std::cout << "Test3" << std::endl;
    //std::cout << "Before Radix Sort: ";
    //for (int i = 0; i < 9; ++i) {
    //    std::cout << input_array3[i] << " ";
    //}
    //std::cout << std::endl;

    //radix_sort_tool.Sort(input_array3, 2, 5, DESCENDING, MostSignificantDigital);
    //std::cout << "After Radix Sort: ";
    //for (int i = 0; i < 9; ++i) {
    //    std::cout << input_array3[i] << " ";
    //}
    //std::cout << std::endl << std::endl;

    system("pause");
    return 0;
}