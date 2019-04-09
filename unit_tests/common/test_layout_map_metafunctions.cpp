/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "gtest/gtest.h"

#include <type_traits>

#include <gridtools/common/gt_assert.hpp>
#include <gridtools/common/layout_map_metafunctions.hpp>

using namespace gridtools;

TEST(layout_map_metafunctions, extend_layout_map) {
    {
        using layout_map_t = layout_map<0, 1, 2, 3>;
        using extended_layout_map_t = extend_layout_map<layout_map_t, 3>::type;
        GT_STATIC_ASSERT((std::is_same<extended_layout_map_t, layout_map<3, 4, 5, 6, 0, 1, 2>>::value), "Error");
    }
    {
        using layout_map_t = layout_map<3, 2, 1, 0>;
        using extended_layout_map_t = extend_layout_map<layout_map_t, 3>::type;
        GT_STATIC_ASSERT((std::is_same<extended_layout_map_t, layout_map<6, 5, 4, 3, 0, 1, 2>>::value), "Error");
    }
    {
        using layout_map_t = layout_map<3, 1, 0, 2>;
        using extended_layout_map_t = extend_layout_map<layout_map_t, 3>::type;
        GT_STATIC_ASSERT((std::is_same<extended_layout_map_t, layout_map<6, 4, 3, 5, 0, 1, 2>>::value), "Error");
    }
}
