/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/storage/common/storage_traits_metafunctions.hpp>

#include "gtest/gtest.h"

#include <type_traits>

using namespace gridtools;

template <class Layout, class Selector, class Expected>
constexpr bool testee = std::is_same<typename get_special_layout<Layout, Selector>::type, Expected>::value;

// 3D
static_assert(testee<layout_map<2, 1, 0>, selector<1, 0, 0>, layout_map<0, -1, -1>>, "");
static_assert(testee<layout_map<2, 1, 0>, selector<0, 1, 0>, layout_map<-1, 0, -1>>, "");
static_assert(testee<layout_map<2, 1, 0>, selector<0, 0, 1>, layout_map<-1, -1, 0>>, "");

static_assert(testee<layout_map<2, 1, 0>, selector<1, 1, 0>, layout_map<1, 0, -1>>, "");
static_assert(testee<layout_map<2, 1, 0>, selector<0, 1, 1>, layout_map<-1, 1, 0>>, "");
static_assert(testee<layout_map<2, 1, 0>, selector<1, 0, 1>, layout_map<1, -1, 0>>, "");

static_assert(testee<layout_map<2, 1, 0>, selector<1, 1, 1>, layout_map<2, 1, 0>>, "");

// 4D
static_assert(testee<layout_map<3, 2, 1, 0>, selector<0, 0, 0, 1>, layout_map<-1, -1, -1, 0>>, "");
static_assert(testee<layout_map<3, 2, 1, 0>, selector<0, 0, 1, 0>, layout_map<-1, -1, 0, -1>>, "");
static_assert(testee<layout_map<3, 2, 1, 0>, selector<0, 1, 0, 0>, layout_map<-1, 0, -1, -1>>, "");
static_assert(testee<layout_map<3, 2, 1, 0>, selector<1, 0, 0, 0>, layout_map<0, -1, -1, -1>>, "");

static_assert(testee<layout_map<3, 2, 1, 0>, selector<0, 0, 1, 1>, layout_map<-1, -1, 1, 0>>, "");
static_assert(testee<layout_map<3, 2, 1, 0>, selector<0, 1, 1, 0>, layout_map<-1, 1, 0, -1>>, "");
static_assert(testee<layout_map<3, 2, 1, 0>, selector<1, 1, 0, 0>, layout_map<1, 0, -1, -1>>, "");
static_assert(testee<layout_map<3, 2, 1, 0>, selector<1, 0, 0, 1>, layout_map<1, -1, -1, 0>>, "");
static_assert(testee<layout_map<3, 2, 1, 0>, selector<0, 1, 0, 1>, layout_map<-1, 1, -1, 0>>, "");
static_assert(testee<layout_map<3, 2, 1, 0>, selector<1, 0, 1, 0>, layout_map<1, -1, 0, -1>>, "");

static_assert(testee<layout_map<3, 2, 1, 0>, selector<0, 1, 1, 1>, layout_map<-1, 2, 1, 0>>, "");
static_assert(testee<layout_map<3, 2, 1, 0>, selector<1, 1, 1, 0>, layout_map<2, 1, 0, -1>>, "");
static_assert(testee<layout_map<3, 2, 1, 0>, selector<1, 1, 0, 1>, layout_map<2, 1, -1, 0>>, "");
static_assert(testee<layout_map<3, 2, 1, 0>, selector<1, 0, 1, 1>, layout_map<2, -1, 1, 0>>, "");

static_assert(testee<layout_map<3, 2, 1, 0>, selector<1, 1, 1, 1>, layout_map<3, 2, 1, 0>>, "");

// 3D
static_assert(testee<layout_map<0, 1, 2>, selector<1, 0, 0>, layout_map<0, -1, -1>>, "");
static_assert(testee<layout_map<0, 1, 2>, selector<0, 1, 0>, layout_map<-1, 0, -1>>, "");
static_assert(testee<layout_map<0, 1, 2>, selector<0, 0, 1>, layout_map<-1, -1, 0>>, "");

static_assert(testee<layout_map<0, 1, 2>, selector<1, 1, 0>, layout_map<0, 1, -1>>, "");
static_assert(testee<layout_map<0, 1, 2>, selector<0, 1, 1>, layout_map<-1, 0, 1>>, "");
static_assert(testee<layout_map<0, 1, 2>, selector<1, 0, 1>, layout_map<0, -1, 1>>, "");

static_assert(testee<layout_map<0, 1, 2>, selector<1, 1, 1>, layout_map<0, 1, 2>>, "");

// 4D
static_assert(testee<layout_map<1, 2, 3, 0>, selector<0, 0, 0, 1>, layout_map<-1, -1, -1, 0>>, "");
static_assert(testee<layout_map<1, 2, 3, 0>, selector<0, 0, 1, 0>, layout_map<-1, -1, 0, -1>>, "");
static_assert(testee<layout_map<1, 2, 3, 0>, selector<0, 1, 0, 0>, layout_map<-1, 0, -1, -1>>, "");
static_assert(testee<layout_map<1, 2, 3, 0>, selector<1, 0, 0, 0>, layout_map<0, -1, -1, -1>>, "");

static_assert(testee<layout_map<1, 2, 3, 0>, selector<0, 0, 1, 1>, layout_map<-1, -1, 1, 0>>, "");
static_assert(testee<layout_map<1, 2, 3, 0>, selector<0, 1, 1, 0>, layout_map<-1, 0, 1, -1>>, "");
static_assert(testee<layout_map<1, 2, 3, 0>, selector<1, 1, 0, 0>, layout_map<0, 1, -1, -1>>, "");
static_assert(testee<layout_map<1, 2, 3, 0>, selector<1, 0, 0, 1>, layout_map<1, -1, -1, 0>>, "");
static_assert(testee<layout_map<1, 2, 3, 0>, selector<0, 1, 0, 1>, layout_map<-1, 1, -1, 0>>, "");
static_assert(testee<layout_map<1, 2, 3, 0>, selector<1, 0, 1, 0>, layout_map<0, -1, 1, -1>>, "");

static_assert(testee<layout_map<1, 2, 3, 0>, selector<0, 1, 1, 1>, layout_map<-1, 1, 2, 0>>, "");
static_assert(testee<layout_map<1, 2, 3, 0>, selector<1, 1, 1, 0>, layout_map<0, 1, 2, -1>>, "");
static_assert(testee<layout_map<1, 2, 3, 0>, selector<1, 1, 0, 1>, layout_map<1, 2, -1, 0>>, "");
static_assert(testee<layout_map<1, 2, 3, 0>, selector<1, 0, 1, 1>, layout_map<1, -1, 2, 0>>, "");

static_assert(testee<layout_map<1, 2, 3, 0>, selector<1, 1, 1, 1>, layout_map<1, 2, 3, 0>>, "");

TEST(dummy, dummy) {}
