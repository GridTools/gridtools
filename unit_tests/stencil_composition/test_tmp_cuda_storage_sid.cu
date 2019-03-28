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
#include <gridtools/meta/list.hpp>
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

        using extent_i_minus = integral_constant<int_t, 1>;
        using extent_i_plus = integral_constant<int_t, 2>;
        using extent_i = meta::list<extent_i_minus, extent_i_plus>;
        using extent_j_minus = integral_constant<int_t, 3>;
        using extent_j_plus = integral_constant<int_t, 4>;
        using extent_j = meta::list<extent_j_minus, extent_j_plus>;
        using extents_t = meta::list<extent_i, extent_j>;

        using blocksize_i = integral_constant<int_t, 32>;
        using blocksize_j = integral_constant<int_t, 8>;
        using blocksizes_t = meta::list<blocksize_i, blocksize_j>;

        using tmp_storage_cuda_t = tmp_storage_cuda<data_t, extents_t, blocksizes_t>;

        TEST(tmp_cuda_storage_sid, concept) {
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

            int expected_stride0 = 1;
            int expected_stride1 = blocksize_i{} + extent_i_minus{} + extent_i_plus{};
            int expected_stride2 = expected_stride1 * (blocksize_j{} + extent_j_minus{} + extent_j_plus{});
            int expected_stride3 = expected_stride2 * n_blocks_i;
            int expected_stride4 = expected_stride3 * n_blocks_j;

            EXPECT_EQ(expected_stride0, get<0>(strides));
            EXPECT_EQ(expected_stride1, get<1>(strides));
            EXPECT_EQ(expected_stride2, get<2>(strides));
            EXPECT_EQ(expected_stride3, get<3>(strides));
            EXPECT_EQ(expected_stride4, get<4>(strides));
        }

        // TODO move allocator into separate file?
        __global__ void test_allocated(float_type *data) { *data = 1; }

        TEST(simple_cuda_allocator, test) {
            // TODO use test functionality
            simple_cuda_allocator alloc;
            auto shared_cuda_ptr = alloc.allocate(sizeof(float_type));

            float_type *ptr = static_cast<float_type *>(shared_cuda_ptr.get());
            float_type data;

            test_allocated<<<1, 1>>>(ptr);
            cudaMemcpy(&data, ptr, sizeof(float_type), cudaMemcpyDeviceToHost);
            ASSERT_EQ(1, data);
        }
    } // namespace
} // namespace gridtools
