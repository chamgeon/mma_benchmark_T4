#pragma once

#include <cstdlib>
#include <cstdio>
#include <type_traits>

// Square problem sizes each CuTe runner is instantiated for (M = N = K = size).
// Every size is a multiple of all kernels' CTA tiles, so no predication is needed.
// Sizes stay compile-time so each instantiation keeps fully static gmem layouts.
template <class F>
void dispatch_size(int argc, char** argv, F&& f){
    int size = argc > 1 ? atoi(argv[1]) : 8192;
    switch(size){
        case 1024: f(std::integral_constant<int, 1024>{}); break;
        case 2048: f(std::integral_constant<int, 2048>{}); break;
        case 4096: f(std::integral_constant<int, 4096>{}); break;
        case 8192: f(std::integral_constant<int, 8192>{}); break;
        default:
            printf("Unsupported size: %d (supported: 1024, 2048, 4096, 8192)\n", size);
            exit(1);
    }
}
