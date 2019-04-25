/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "test_tmp_storage_sid_cuda.cpp"

#include "../../test_helper.hpp"
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/stencil_composition/backend_cuda/simple_device_memory_allocator.hpp>

namespace gridtools {
    namespace {
        using tmp_cuda_storage_sid_float = tmp_cuda_storage_sid<float_type, simple_device_memory_allocator>;
        TEST_F(tmp_cuda_storage_sid_float, maker_with_device_allocator) {
            auto testee = get_tmp_storage();

            using tmp_cuda_t = decltype(testee);

            static_assert(sid::is_sid<tmp_cuda_t>::value, "is_sid()");

            ASSERT_TYPE_EQ<GT_META_CALL(sid::ptr_type, tmp_cuda_t), data_t *>();
            ASSERT_TYPE_EQ<GT_META_CALL(sid::ptr_diff_type, tmp_cuda_t), int_t>();
        }
    } // namespace
} // namespace gridtools
