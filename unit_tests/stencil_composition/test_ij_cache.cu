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
    __device__ ptrdiff_t get_origin_offset(ptr_holder<float_type> testee) {
        extern __shared__ float_type shm[];
        return reinterpret_cast<float_type *>(testee()) - shm;
    }

    namespace {
        namespace tu = tuple_util;
        using tuple_util::get;

        static constexpr int i_size = 9;
        static constexpr int j_size = 13;
        static constexpr int i_zero = 5;
        static constexpr int j_zero = 3;

        template <typename T>
        using ij_cache_t = sid_ij_cache<T, i_size, j_size, i_zero, j_zero>;

        TEST(sid_ij_cache, smoke) {

            shared_allocator allocator;
            ij_cache_t<float_type> testee{allocator};

            static_assert(is_sid<ij_cache_t<float_type>>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_type, ij_cache_t<float_type>), float_type *>(), "");
            static_assert(std::is_same<GT_META_CALL(sid::ptr_diff_type, ij_cache_t<float_type>), std::ptrdiff_t>(), "");

            using expected_kind = hymap::keys<dim::i, dim::j>::values<gridtools::integral_constant<int_t, 1>,
                gridtools::integral_constant<int_t, i_size>>;
            static_assert(std::is_same<GT_META_CALL(sid::strides_kind, ij_cache_t<float_type>), expected_kind>(), "");

            auto strides = sid::get_strides(testee);
            EXPECT_EQ(1, at_key<dim::i>(strides));
            EXPECT_EQ(i_size, at_key<dim::j>(strides));

            EXPECT_EQ(1, sid::get_stride<dim::i>(strides));
            EXPECT_EQ(i_size, sid::get_stride<dim::j>(strides));
            EXPECT_EQ(0, sid::get_stride<dim::k>(strides));

            ij_cache_t<float_type> another_testee{allocator};

            int_t offset = exec_with_shared_memory(allocator.size(), MAKE_CONSTANT(get_origin_offset), sid::get_origin(testee));
            int_t offset2 = exec_with_shared_memory(allocator.size(), MAKE_CONSTANT(get_origin_offset), sid::get_origin(another_testee));

            // the first offset should be large enough to fit the zero offsets
            EXPECT_EQ(offset, i_size * j_zero + i_zero);

            // between the two offsets, we should have enough space to fit one ij cache
            EXPECT_LE(offset2, offset + i_size * j_size);

            // between the second offset and the size of the buffer, we should have enough space to fit the buffer
            EXPECT_EQ(allocator.size() / sizeof(float_type) - offset2, i_size * j_size - i_size * j_zero - i_zero);
        }
    } // namespace
} // namespace gridtools
