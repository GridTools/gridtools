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

#include <gridtools/stencil_composition/backend_cuda/simple_device_memory_allocator.hpp>
#include <gridtools/stencil_composition/sid/concept.hpp>
#include <gridtools/tools/backend_select.hpp>

#include "../test_helper.hpp"
#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        TEST(tmp_cuda_storage, maker_with_device_allocator) {
            simple_device_memory_allocator alloc;

            // For CUDA 8.0 we need to instantiate the type, otherwise is_sid will fail.
#ifndef GT_ICOSAHEDRAL_GRIDS
            auto testee = make_tmp_storage_cuda<float_type>(tmp_cuda::blocksize<1, 1>{}, extent<>{}, 0, 0, 0, alloc);
#else
            auto testee = make_tmp_storage_cuda<float_type>(
                tmp_cuda::blocksize<1, 1>{}, extent<>{}, color_type<1>{}, 0, 0, 0, alloc);
#endif
            using testee_t = decltype(testee);

            static_assert(sid::is_sid<testee_t>::value, "is_sid()");
#if !(defined(__CUDACC__) && defined(__clang__) && __CUDACC_VER_MAJOR__ < 10)
            // fails with internal error in cudafe++ for CUDA < 10 and clang as host compiler
            ASSERT_TYPE_EQ<sid::ptr_type<testee_t>, float_type *>();
            ASSERT_TYPE_EQ<sid::ptr_diff_type<testee_t>, int_t>();
#endif
        }

    } // namespace
} // namespace gridtools

#include "test_tmp_storage_sid_cuda.cpp"
