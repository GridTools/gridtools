/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/positional.hpp>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/dim.hpp>
#include <gridtools/stencil_composition/sid/concept.hpp>

namespace gridtools {
    namespace {
        TEST(positional, smoke) {
            auto testee = make_positional_sid({1, 2, 3});

            auto ptr = sid::get_origin(testee)();

            EXPECT_EQ((*ptr).i, 1);
            EXPECT_EQ((*ptr).j, 2);
            EXPECT_EQ((*ptr).k, 3);

            auto strides = sid::get_strides(testee);

            sid::shift(ptr, sid::get_stride<dim::i>(strides), -34);
            sid::shift(ptr, sid::get_stride<dim::j>(strides), 8);
            sid::shift(ptr, sid::get_stride<dim::k>(strides), 11);

            EXPECT_EQ((*ptr).i, -33);
            EXPECT_EQ((*ptr).j, 10);
            EXPECT_EQ((*ptr).k, 14);

            using diff_t = GT_META_CALL(sid::ptr_diff_type, positional_sid_t);

            diff_t diff{};

            sid::shift(diff, sid::get_stride<dim::i>(strides), -34);
            sid::shift(diff, sid::get_stride<dim::j>(strides), 8);
            sid::shift(diff, sid::get_stride<dim::k>(strides), 11);

            ptr = sid::get_origin(testee)() + diff;

            EXPECT_EQ((*ptr).i, -33);
            EXPECT_EQ((*ptr).j, 10);
            EXPECT_EQ((*ptr).k, 14);
        }
    } // namespace
} // namespace gridtools
