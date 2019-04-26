/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/sid/block.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/hymap.hpp>
#include <gridtools/stencil_composition/dim.hpp>
#include <gridtools/stencil_composition/positional.hpp>

namespace gridtools {
    namespace {
        struct unused_dim;

        TEST(sid_block, smoke) {
            const int domain_size_i = 12;
            const int domain_size_j = 14;
            constexpr int block_size_i = 3;
            const int block_size_j = 7;

            auto blocks = tuple_util::make<hymap::keys<dim::i, dim::j, unused_dim>::values>(
                integral_constant<int_t, block_size_i>{}, block_size_j, 5);

            positional s{0, 0, 0};
            auto blocked_s = sid::block(s, blocks);
            static_assert(is_sid<decltype(blocked_s)>(), "");

            auto strides = sid::get_strides(blocked_s);
            for (int ib = 0; ib < domain_size_i / block_size_i; ++ib) {
                for (int jb = 0; jb < domain_size_j / block_size_j; ++jb) {
                    auto ptr = sid::get_origin(blocked_s);
                    sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::i>>(strides), ib);
                    sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::j>>(strides), jb);

                    for (int i = ib * block_size_i; i < (ib + 1) * block_size_i; ++i) {
                        for (int j = jb * block_size_j; j < (jb + 1) * block_size_j; ++j) {
                            EXPECT_EQ((*ptr).i, i);
                            EXPECT_EQ((*ptr).j, j);
                            EXPECT_EQ((*ptr).k, 0);
                            sid::shift(ptr, sid::get_stride<dim::j>(strides), 1);
                        }
                        sid::shift(ptr, sid::get_stride<dim::j>(strides), -block_size_j);
                        sid::shift(ptr, sid::get_stride<dim::i>(strides), 1);
                    }
                }
            }
        }
    } // namespace
} // namespace gridtools
