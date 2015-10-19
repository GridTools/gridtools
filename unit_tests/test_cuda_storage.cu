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
#include "common/gpu_clone.hpp"
#include "storage/hybrid_pointer.hpp"
#include "stencil-composition/backend.hpp"
#include "common/layout_map.hpp"
#include "common/defs.hpp"
#include "stencil-composition/backend.hpp"

using gridtools::uint_t;
using gridtools::int_t;

#ifdef __CUDACC__
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
#endif

using namespace gridtools;
using namespace enumtype;
bool test_cuda_storage() {

    typedef gridtools::backend<gridtools::enumtype::Cuda, gridtools::enumtype::Block > backend_t;
    typedef gridtools::backend<Cuda, Block>::storage_type<float_type, storage_info<0,layout_map<0,1,2> > > ::type storage_type;

    uint_t d1 = 3;
    uint_t d2 = 3;
    uint_t d3 = 3;

    typename storage_type::meta_data_t meta_(d1,d2,d3);
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
    meta_.clone_to_device();//copy meta information to the GPU
#ifdef __CUDACC__
    add_on_gpu<<<1,1>>>(meta_.gpu_object_ptr, data.gpu_object_ptr, d1, d2, d3);
    cudaDeviceSynchronize();
#endif
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
