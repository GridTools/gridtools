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
            const int domain_size_k = 4;
            constexpr int block_size_i = 3;
            const int block_size_j = 7;

            auto blocks = tuple_util::make<hymap::keys<dim::i, dim::j, unused_dim>::values>(
                integral_constant<int_t, block_size_i>{}, block_size_j, 5);

            positional s{0, 0, 0};
            auto blocked_s = sid::block(s, blocks);
            static_assert(is_sid<decltype(blocked_s)>(), "");

            auto strides = sid::get_strides(blocked_s);
            for (int ib = 0; ib < domain_size_i; ib += block_size_i) {
                for (int jb = 0; jb < domain_size_j; jb += block_size_j) {
                    auto ptr = sid::get_origin(blocked_s);
                    sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::i>>(strides), ib / block_size_i);
                    sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::j>>(strides), jb / block_size_j);

                    for (int i = ib; i < ib + block_size_i; ++i) {
                        for (int j = jb; j < jb + block_size_j; ++j) {
                            for (int k = 0; k < domain_size_k; ++k) {
                                EXPECT_EQ((*ptr).i, i);
                                EXPECT_EQ((*ptr).j, j);
                                EXPECT_EQ((*ptr).k, k);
                                sid::shift(ptr, sid::get_stride<dim::k>(strides), 1);
                            }
                            sid::shift(ptr, sid::get_stride<dim::k>(strides), -domain_size_k);
                            sid::shift(ptr, sid::get_stride<dim::j>(strides), 1);
                        }
                        sid::shift(ptr, sid::get_stride<dim::j>(strides), -block_size_j);
                        sid::shift(ptr, sid::get_stride<dim::i>(strides), 1);
                    }
                }
            }
        }

        TEST(sid_block, multilevel) {
            positional s{0, 0, 0};

            const int domain_size_i = 20;
            const int block_size_1 = 5;
            const int block_size_2 = 2;
            auto blocked_s = sid::block(s, tuple_util::make<hymap::keys<dim::i>::values>(block_size_1));
            static_assert(is_sid<decltype(blocked_s)>(), "");
            auto blocked_blocked_s =
                sid::block(blocked_s, tuple_util::make<hymap::keys<sid::blocked_dim<dim::i>>::values>(block_size_2));
            static_assert(is_sid<decltype(blocked_blocked_s)>(), "");

            auto ptr = sid::get_origin(blocked_blocked_s);
            auto strides = sid::get_strides(blocked_blocked_s);
            for (int ib2 = 0; ib2 < domain_size_i; ib2 += block_size_1 * block_size_2) {
                for (int ib = ib2; ib < ib2 + block_size_1 * block_size_2; ib += block_size_1) {
                    for (int i = ib; i < ib + block_size_1; ++i) {
                        EXPECT_EQ((*ptr).i, i);
                        EXPECT_EQ((*ptr).j, 0);
                        EXPECT_EQ((*ptr).k, 0);
                        sid::shift(ptr, sid::get_stride<dim::i>(strides), 1);
                    }
                    sid::shift(ptr, sid::get_stride<dim::i>(strides), -block_size_1);
                    sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::i>>(strides), 1);
                }
                sid::shift(ptr, sid::get_stride<sid::blocked_dim<dim::i>>(strides), -block_size_2);
                sid::shift(ptr, sid::get_stride<sid::blocked_dim<sid::blocked_dim<dim::i>>>(strides), 1);
            }
        }

        TEST(sid_block, do_nothing) {
            positional s{0, 0, 0};

            auto same_s = sid::block(s, tuple_util::make<hymap::keys<unused_dim>::values>(42));
            static_assert(std::is_same<decltype(s), decltype(same_s)>(), "");
        }
    } // namespace
} // namespace gridtools
