/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/fn/extents.hpp>

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/integral_constant.hpp>
#include <gridtools/common/tuple_util.hpp>

#include <cuda_test_helper.hpp>

namespace gridtools::fn {
    namespace {
        struct a;
        struct b;
        struct c;

        template <class Extent>
        __device__ hymap::keys<a, b, c>::values<long int, int, long int> extend_offsets_device(
            hymap::keys<a, b, c>::values<int, int, int> const &offsets) {
            return extend_offsets<Extent>(offsets);
        }

        TEST(extend_offsets, device) {
            using ext = extents<extent<a, -1, 0>, extent<b, 0, 2>, extent<c, 1, 1>>;
            auto offsets = tuple_util::make<hymap::keys<a, b, c>::values>(0, 1, 2);

            auto testee = on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&extend_offsets_device<ext>), offsets);

            ASSERT_EQ(at_key<a>(testee), -1);
            ASSERT_EQ(at_key<b>(testee), 1);
            ASSERT_EQ(at_key<c>(testee), 3);
        }

        template <class Extent>
        __device__ hymap::keys<a, b, c>::values<long unsigned int, long unsigned int, int> extend_sizes_device(
            hymap::keys<a, b, c>::values<int, int, int> const &sizes) {
            return extend_sizes<Extent>(sizes);
        }

        TEST(extend_sizes, device) {
            using ext = extents<extent<a, -1, 0>, extent<b, 0, 2>, extent<c, 1, 1>>;
            auto sizes = tuple_util::make<hymap::keys<a, b, c>::values>(4, 5, 6);

            auto testee = on_device::exec(GT_MAKE_INTEGRAL_CONSTANT_FROM_VALUE(&extend_sizes_device<ext>), sizes);

            ASSERT_EQ(at_key<a>(testee), 5);
            ASSERT_EQ(at_key<b>(testee), 7);
            ASSERT_EQ(at_key<c>(testee), 6);
        }

    } // namespace
} // namespace gridtools::fn

#include "test_extents.cpp"
