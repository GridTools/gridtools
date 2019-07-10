/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/backend_cuda/tmp_storage_sid.hpp>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/backend_cuda/simple_device_memory_allocator.hpp>
#include <gridtools/stencil_composition/color.hpp>
#include <gridtools/stencil_composition/dim.hpp>
#include <gridtools/stencil_composition/extent.hpp>
#include <gridtools/stencil_composition/sid/concept.hpp>

#include "../cuda_test_helper.hpp"

namespace gridtools {
    namespace on_device {

        namespace {
            using extent_t = extent<>;
            using block_size_t = tmp_cuda::blocksize<2, 2>;

            struct smoke_f {
                template <class PtrHolder, class Strides>
                __host__ __device__ bool operator()(PtrHolder const &holder, Strides const &strides) const {
                    auto ptr = holder();
                    sid::shift(ptr, sid::get_stride<dim::i>(strides), 1);
                    sid::shift(ptr, sid::get_stride<dim::j>(strides), 1);
                    sid::shift(ptr, sid::get_stride<dim::k>(strides), 1);
                    *ptr = 42;
                    return *ptr == 42;
                }
            };

            TEST(tmp_cuda_storage, maker_with_device_allocator) {
                simple_device_memory_allocator alloc;
                auto testee = make_tmp_storage_cuda<int>(block_size_t{},
                    extent_t{},
#ifdef GT_ICOSAHEDRAL_GRIDS
                    color_type<1>{},
#endif
                    1,
                    1,
                    2,
                    alloc);
                EXPECT_TRUE(exec(smoke_f{}, sid::get_origin(testee), sid::get_strides(testee)));
            }
        } // namespace
    }     // namespace on_device
} // namespace gridtools

#include "test_tmp_storage_sid_cuda.cpp"
