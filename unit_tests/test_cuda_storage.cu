/*
 * File:   test_domain.cpp
 * Author: mbianco
 *
 * Created on February 14, 2014, 4:18 PM
 *
 * Test cuda_storage features
 */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <common/gpu_clone.h>
#include <storage/hybrid_pointer.h>
#include <storage/cuda_storage.h>
#include <common/layout_map.h>

#ifdef __CUDACC__
template <typename T>
__global__
void add_on_gpu(T * ptr, int d1, int d2, int d3) {
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d3; ++k) {
                (*ptr)(i,j,k) = -i-j-k;
            }
        }
    }
}
#endif

bool test_cuda_storage() {

    typedef gridtools::base_storage<gridtools::enumtype::Cuda, double, gridtools::layout_map<0,1,2> > storage_type;

    int d1 = 3;
    int d2 = 3;
    int d3 = 3;

    storage_type data(d1,d2,d3,-1, std::string("data"));

    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d3; ++k) {
                data(i,j,k) = i+j+k;
#ifndef NDEBUG
                std::cout << data(i,j,k) << " ";
#endif
            }
#ifndef NDEBUG
            std::cout << std::endl;
#endif
        }
#ifndef NDEBUG
        std::cout << std::endl;
        std::cout << std::endl;
#endif
    }

    data.h2d_update();
    data.clone_to_gpu();

#ifdef __CUDACC__
    add_on_gpu<<<1,1>>>(data.gpu_object_ptr, d1, d2, d3);
    cudaDeviceSynchronize();
#endif
    data.d2h_update();

    bool same = true;
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            for (int k = 0; k < d3; ++k) {
#ifndef NDEBUG
                std::cout << data(i,j,k) << " ";
#endif
                if (data(i,j,k) != -i-j-k)
                    same = false;
            }
#ifndef NDEBUG
            std::cout << std::endl;
#endif
        }
#ifndef NDEBUG
        std::cout << std::endl;
        std::cout << std::endl;
#endif
    }

    return same;
}
