/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/common/tuple_util.hpp>
#include <gridtools/stencil_composition/backend_cuda/tmp_storage_sid.hpp>
#include <gridtools/stencil_composition/sid/concept.hpp>
#include <gridtools/tools/backend_select.hpp>

#include <type_traits>

#include "../test_helper.hpp"
#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        namespace tu = tuple_util;
        using tuple_util::get;

        using data_t = float_type;
        using extent_t = array<int_t, 2>;
        using extents_t = array<extent_t, 2>;
        using blocksize_i = integral_constant<int_t, 32>;
        using blocksize_j = integral_constant<int_t, 8>;
        using blocksizes_t = tuple<blocksize_i, blocksize_j>;
        using tmp_storage_cuda_t = tmp_storage_cuda<data_t, extents_t, blocksizes_t>;
        TEST(tmp_cuda_storage_sid, smoke) {
            static_assert(is_sid<tmp_storage_cuda_t>(), "");
            ASSERT_TYPE_EQ<GT_META_CALL(sid::ptr_type, tmp_storage_cuda_t), data_t *>();
            ASSERT_TYPE_EQ<GT_META_CALL(sid::ptr_diff_type, tmp_storage_cuda_t), int_t>();
            static_assert(
                std::is_same<GT_META_CALL(sid::strides_kind, tmp_storage_cuda_t), tmp_cuda_strides_kind<extents_t>>(),
                "");

            using strides_t = GT_META_CALL(sid::strides_type, tmp_storage_cuda_t);

            static_assert(tu::size<strides_t>() == 5, "");
            ASSERT_TYPE_EQ<GT_META_CALL(tu::element, (0, strides_t)), integral_constant<int_t, 1>>();
            ASSERT_TYPE_EQ<GT_META_CALL(tu::element, (3, strides_t)), int_t>();
            ASSERT_TYPE_EQ<GT_META_CALL(tu::element, (4, strides_t)), int_t>();
        }

        TEST(tmp_cuda_storage_sid, strides) {
            int_t n_blocks_i = 11;
            int_t n_blocks_j = 12;
            int_t k_size = 13;
            tmp_storage_cuda_t testee = {{n_blocks_i, n_blocks_j}, k_size};

            EXPECT_EQ(testee.m_cuda_ptr.get(), sid::get_origin(testee)());

            auto strides = sid::get_strides(testee);
            auto expected_strides = array<int, 5>{1, //
                blocksize_i{},
                blocksize_i{} * blocksize_j{},
                blocksize_i{} * blocksize_j{} * n_blocks_i,
                blocksize_i{} * blocksize_j{} * n_blocks_i * n_blocks_j};

            EXPECT_EQ(get<0>(expected_strides), get<0>(strides));
            EXPECT_EQ(get<1>(expected_strides), get<1>(strides));
            EXPECT_EQ(get<2>(expected_strides), get<2>(strides));
            EXPECT_EQ(get<3>(expected_strides), get<3>(strides));
            EXPECT_EQ(get<4>(expected_strides), get<4>(strides));
        }

        //        __global__ void test_allocated(float_type *data) { *data = 1; }
        //
        //        TEST(simple_cuda_allocator, test) {
        //            simple_cuda_allocator alloc;
        //            auto shared_cuda_ptr = alloc.allocate(sizeof(float_type));
        //        }
    } // namespace
} // namespace gridtools
