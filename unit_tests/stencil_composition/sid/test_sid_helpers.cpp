/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/sid/sid_helpers.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/hymap.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/stencil_composition/dim.hpp>
#include <gridtools/stencil_composition/sid/synthetic.hpp>

namespace gridtools {
    namespace {
        using sid::property;
        using namespace literals;
        namespace tu = tuple_util;

        TEST(sid_helpers, shifted) {
            double data[3][5][7];

            auto src = sid::synthetic()
                           .set<property::origin>(sid::host_device::make_simple_ptr_holder(&data[0][0][0]))
                           .set<property::strides>(tuple_util::make<tuple>(5_c * 7_c, 7_c, 1_c));

            auto offset = tuple_util::make<tuple>(1_c, 2);
            auto testee = sid::shifted_sid(src, offset);

            static_assert(is_sid<decltype(testee)>(), "");

            EXPECT_EQ(&data[0][0][0], sid::get_origin(src)());
            EXPECT_EQ(&data[1][2][0], sid::get_origin(testee)());
        }
    } // namespace
} // namespace gridtools
