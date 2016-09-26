/*
 * File:   test_domain.cpp
 * Author: mbianco
 *
 * Created on February 14, 2014, 4:18 PM
 *
 * Test cuda_storage features
 */
#include "gtest/gtest.h"

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <stencil-composition/stencil-composition.hpp>

using gridtools::uint_t;
using gridtools::int_t;

template <typename T, typename U>
__global__
void add_on_gpu(U* meta, T * ptr, uint_t d1, uint_t d2, uint_t d3) {
    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
                ptr->fields_view()[0][meta->index(i,j,k)] = -i-j-k;
            }
        }
    }
}

using namespace gridtools;
using namespace enumtype;
bool test_cuda_storage() {

    typedef backend<Cuda, GRIDBACKEND, Block > backend_t;
    typedef backend_t::storage_type<float_type, backend_t::storage_info<0,layout_map<0,1,2> > > ::type storage_type;

    uint_t d1 = 3;
    uint_t d2 = 3;
    uint_t d3 = 3;

    typename storage_type::storage_info_type meta_(d1,d2,d3);
    storage_type data(meta_, -1., "data"); //allocate on GPU

    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
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

    data.h2d_update(); //copy to GPU
    data.clone_to_device();

    // clang-format off
    add_on_gpu<<<1,1>>>(data.get_meta_data_pointer().get(), data.get_pointer_to_use(), d1, d2, d3);
    // clang-format on
    cudaDeviceSynchronize();

    data.d2h_update();

    bool same = true;
    for (uint_t i = 0; i < d1; ++i) {
        for (uint_t j = 0; j < d2; ++j) {
            for (uint_t k = 0; k < d3; ++k) {
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

TEST(test_cudastorage, functionality_test) {
    EXPECT_EQ(test_cuda_storage(), true);
}
