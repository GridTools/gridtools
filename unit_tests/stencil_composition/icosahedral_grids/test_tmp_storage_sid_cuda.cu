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
#include <gridtools/stencil_composition/backend_cuda/simple_device_memory_allocator.hpp>
#include <gridtools/stencil_composition/backend_cuda/tmp_storage_sid.hpp>
#include <gridtools/stencil_composition/color.hpp>
#include <gridtools/stencil_composition/sid/concept.hpp>
#include <gridtools/tools/backend_select.hpp>

#include <type_traits>

#include "../../cuda_test_helper.hpp"
#include "../../test_helper.hpp"
#include <gtest/gtest.h>

namespace gridtools {
    namespace {
        template <typename PtrHolder>
        __device__ float_type *get_ptr(PtrHolder ptr_holder) {
            return ptr_holder();
        }

        namespace tu = tuple_util;
        using tuple_util::get;

        using data_t = float_type;

        constexpr int_t extent_i_minus = -1;
        constexpr int_t extent_i_plus = 2;
        constexpr int_t extent_j_minus = -3;
        constexpr int_t extent_j_plus = 4;

        constexpr int_t blocksize_i = 32;
        constexpr int_t blocksize_j = 8;

        constexpr uint_t n_colors = 2;

        TEST(tmp_cuda_storage_sid, maker) {
            int_t n_blocks_i = 11;
            int_t n_blocks_j = 12;
            int_t k_size = 13;

            simple_device_memory_allocator alloc;

            auto testee = make_tmp_storage_cuda<data_t>(tmp_cuda::blocksize<blocksize_i, blocksize_j>{},
                extent<extent_i_minus, extent_i_plus, extent_j_minus, extent_j_plus>{},
                color_type<n_colors>{},
                n_blocks_i,
                n_blocks_j,
                k_size,
                alloc);

            using tmp_cuda_t = decltype(testee);
            static_assert(is_sid<tmp_cuda_t>(), ""); // TODO error here
            ASSERT_TYPE_EQ<GT_META_CALL(sid::ptr_type, tmp_cuda_t), data_t *>();
            ASSERT_TYPE_EQ<GT_META_CALL(sid::ptr_diff_type, tmp_cuda_t), int_t>();

            auto strides = sid::get_strides(testee);

            int expected_stride0 = 1;
            int expected_stride1 = blocksize_i - extent_i_minus + extent_i_plus;
            int expected_stride2 = expected_stride1 * (blocksize_j - extent_j_minus + extent_j_plus);
            int expected_stride3 = expected_stride2 * n_colors;
            int expected_stride4 = expected_stride3 * n_blocks_i;
            int expected_stride5 = expected_stride4 * n_blocks_j;

            EXPECT_EQ(expected_stride0, at_key<dim::i>(strides));
            EXPECT_EQ(expected_stride1, at_key<dim::j>(strides));
            EXPECT_EQ(expected_stride2, at_key<dim::c>(strides));
            EXPECT_EQ(expected_stride3, at_key<tmp_cuda::block_i>(strides));
            EXPECT_EQ(expected_stride4, at_key<tmp_cuda::block_j>(strides));
            EXPECT_EQ(expected_stride5, at_key<dim::k>(strides));

            auto ptr_to_allocation = static_cast<float_type *>(alloc.ptrs()[0].get());
            auto ptr_to_origin =
                ptr_to_allocation - expected_stride0 * extent_i_minus - expected_stride1 * extent_j_minus;

            auto sid_origin = gridtools::on_device::exec(
                GT_MAKE_CONSTANT(get_ptr<decltype(sid::get_origin(testee))>), sid::get_origin(testee));
            EXPECT_EQ(ptr_to_origin, sid_origin);
        }
    } // namespace
} // namespace gridtools
