/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "../test_helper.hpp"
#include <gridtools/meta.hpp>
#include <gridtools/stencil_composition/backend_cuda/simple_device_memory_allocator.hpp>
#include <gridtools/stencil_composition/backend_cuda/tmp_storage_sid.hpp>
#include <gridtools/stencil_composition/color.hpp>
#include <gridtools/stencil_composition/sid/concept.hpp>
#include <gridtools/tools/backend_select.hpp>

#include <gtest/gtest.h>

namespace gridtools {
    namespace {

#ifndef GT_ICOSAHEDRAL_GRIDS
        using testee_t = decltype(make_tmp_storage_cuda<float_type>(
            tmp_cuda::blocksize<1, 1>{}, extent<>{}, 0, 0, 0, std::declval<simple_device_memory_allocator &>()));
#else
        using testee_t = decltype(make_tmp_storage_cuda<float_type>(tmp_cuda::blocksize<1, 1>{},
            extent<>{},
            color_type<1>{},
            0,
            0,
            0,
            std::declval<simple_device_memory_allocator &>()));
#endif

        static_assert(sid::is_sid<testee_t>::value, "is_sid()");
        TEST(tmp_cuda_storage, maker_with_device_allocator) {
            ASSERT_TYPE_EQ<GT_META_CALL(sid::ptr_type, testee_t), float_type *>();
            ASSERT_TYPE_EQ<GT_META_CALL(sid::ptr_diff_type, testee_t), int_t>();
        }

    } // namespace
} // namespace gridtools

#include "test_tmp_storage_sid_cuda.cpp"
