/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/backend_cuda/ij_cache.hpp>

#include <gridtools/stencil_composition/sid/concept.hpp>
#include <gridtools/tools/backend_select.hpp>

#include <gtest/gtest.h>

#include "../cuda_test_helper.hpp"

namespace gridtools {
    namespace {
        template <typename PtrHolder>
        __device__ ptrdiff_t get_origin_offset(PtrHolder ptr_holder) {
            extern __shared__ typename PtrHolder::element_type shm[];
            return ptr_holder() - shm;
        }

        static constexpr int i_size = 9;
        static constexpr int j_size = 13;
        static constexpr int i_zero = 5;
        static constexpr int j_zero = 3;

        TEST(sid_ij_cache, smoke) {

            shared_allocator allocator;
            auto testee = make_ij_cache<double, i_size, j_size, i_zero, j_zero>(allocator);

            using ij_cache_t = decltype(testee);

            static_assert(is_sid<ij_cache_t>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_type, ij_cache_t), float_type *>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_diff_type, ij_cache_t), std::ptrdiff_t>(), "");

            using expected_kind = hymap::keys<dim::i, dim::j>::values<gridtools::integral_constant<int_t, 1>,
                gridtools::integral_constant<int_t, i_size>>;
            static_assert(std::is_same<GT_META_CALL(sid::strides_kind, ij_cache_t), expected_kind>(), "");

            auto strides = sid::get_strides(testee);
            EXPECT_EQ(1, at_key<dim::i>(strides));
            EXPECT_EQ(i_size, at_key<dim::j>(strides));

            EXPECT_EQ(1, sid::get_stride<dim::i>(strides));
            EXPECT_EQ(i_size, sid::get_stride<dim::j>(strides));
            EXPECT_EQ(0, sid::get_stride<dim::k>(strides));

            auto another_testee = make_ij_cache<double, i_size, j_size, i_zero, j_zero>(allocator);

            auto origin1 = sid::get_origin(testee);
            int_t offset1 = on_device::exec_with_shared_memory(
                allocator.size(), MAKE_CONSTANT(get_origin_offset<decltype(origin1)>), origin1);
            auto origin2 = sid::get_origin(another_testee);
            int_t offset2 = on_device::exec_with_shared_memory(
                allocator.size(), MAKE_CONSTANT(get_origin_offset<decltype(origin2)>), origin2);

            // the first offset should be large enough to fit the zero offsets
            EXPECT_LE(offset1, i_size * j_zero + i_zero);

            // between the two offsets, we should have enough space to fit one ij cache
            EXPECT_LE(offset2, offset1 + i_size * j_size);

            // between the second offset and the size of the buffer, we should have enough space to fit the buffer
            EXPECT_LE(allocator.size() / sizeof(float_type) - offset2, i_size * j_size - i_size * j_zero - i_zero);
        }
    } // namespace
} // namespace gridtools
