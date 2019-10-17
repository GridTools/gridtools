/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/layout_map_metafunctions.hpp>

#include <type_traits>

#include "gtest/gtest.h"

using namespace gridtools;

template <class Layout, class Expected>
constexpr bool testee = std::is_same<typename extend_layout_map<Layout, 3>::type, Expected>::value;

static_assert(testee<layout_map<0, 1, 2, 3>, layout_map<3, 4, 5, 6, 0, 1, 2>>, "");
static_assert(testee<layout_map<3, 2, 1, 0>, layout_map<6, 5, 4, 3, 0, 1, 2>>, "");
static_assert(testee<layout_map<3, 1, 0, 2>, layout_map<6, 4, 3, 5, 0, 1, 2>>, "");

TEST(dummy, dummy) {}
