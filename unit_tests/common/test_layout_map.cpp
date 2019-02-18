/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "gtest/gtest.h"

#include <gridtools/common/gt_assert.hpp>
#include <gridtools/common/layout_map.hpp>
#include <gridtools/common/layout_map_metafunctions.hpp>

#include "../test_helper.hpp"
using namespace gridtools;

template <typename T>
constexpr uint_t get_length() {
    return T::masked_length;
}

TEST(LayoutMap, SimpleLayout) {
    typedef layout_map<0, 1, 2> layout1;

    // test length
    GT_STATIC_ASSERT(layout1::masked_length == 3, "layout_map length is wrong");
    GT_STATIC_ASSERT(layout1::unmasked_length == 3, "layout_map length is wrong");

    // test find method
    GT_STATIC_ASSERT(layout1::find<0>() == 0, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout1::find<1>() == 1, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout1::find<2>() == 2, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout1::find(0) == 0, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout1::find(1) == 1, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout1::find(2) == 2, "wrong result in layout_map find method");

    // test at method
    GT_STATIC_ASSERT(layout1::at<0>() == 0, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout1::at<1>() == 1, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout1::at<2>() == 2, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout1::at(0) == 0, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout1::at(1) == 1, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout1::at(2) == 2, "wrong result in layout_map at method");
}

TEST(LayoutMap, ExtendedLayout) {
    typedef layout_map<3, 2, 1, 0> layout2;

    // test length
    GT_STATIC_ASSERT(layout2::masked_length == 4, "layout_map length is wrong");
    GT_STATIC_ASSERT(layout2::unmasked_length == 4, "layout_map length is wrong");

    // test find method
    GT_STATIC_ASSERT(layout2::find<0>() == 3, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout2::find<1>() == 2, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout2::find<2>() == 1, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout2::find<3>() == 0, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout2::find(0) == 3, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout2::find(1) == 2, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout2::find(2) == 1, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout2::find(3) == 0, "wrong result in layout_map find method");

    // test at method
    GT_STATIC_ASSERT(layout2::at<0>() == 3, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout2::at<1>() == 2, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout2::at<2>() == 1, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout2::at<3>() == 0, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout2::at(0) == 3, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout2::at(1) == 2, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout2::at(2) == 1, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout2::at(3) == 0, "wrong result in layout_map at method");
}

TEST(LayoutMap, MaskedLayout) {
    typedef layout_map<2, -1, 1, 0> layout3;

    // test length
    GT_STATIC_ASSERT(layout3::masked_length == 4, "layout_map length is wrong");
    GT_STATIC_ASSERT(layout3::unmasked_length == 3, "layout_map length is wrong");

    // test find method
    GT_STATIC_ASSERT(layout3::find<0>() == 3, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout3::find<1>() == 2, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout3::find<2>() == 0, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout3::find(0) == 3, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout3::find(1) == 2, "wrong result in layout_map find method");
    GT_STATIC_ASSERT(layout3::find(2) == 0, "wrong result in layout_map find method");

    // test at method
    GT_STATIC_ASSERT(layout3::at<0>() == 2, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout3::at<1>() == -1, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout3::at<2>() == 1, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout3::at<3>() == 0, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout3::at(0) == 2, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout3::at(1) == -1, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout3::at(2) == 1, "wrong result in layout_map at method");
    GT_STATIC_ASSERT(layout3::at(3) == 0, "wrong result in layout_map at method");
}

TEST(LayoutMap, DefaultLayout) {
    ASSERT_TYPE_EQ<layout_map<0>, default_layout_map<1>::type>();
    ASSERT_TYPE_EQ<layout_map<0, 1>, default_layout_map<2>::type>();
    ASSERT_TYPE_EQ<layout_map<0, 1, 2>, default_layout_map<3>::type>();
    ASSERT_TYPE_EQ<layout_map<0, 1, 2, 3>, default_layout_map<4>::type>();

    ASSERT_TYPE_EQ<layout_map<0, 1, 2, 3>, default_layout_map_t<4>>();
}

TEST(LayoutMap, Extender) {
    typedef layout_map<0, 1, 2> layout;

    typedef typename extend_layout_map<layout, 1>::type ext_layout_1;
    ASSERT_TYPE_EQ<layout_map<1, 2, 3, 0>, ext_layout_1>();

    typedef typename extend_layout_map<layout, 2>::type ext_layout_2;
    ASSERT_TYPE_EQ<layout_map<2, 3, 4, 0, 1>, ext_layout_2>();

    typedef typename extend_layout_map<layout, 3>::type ext_layout_3;
    ASSERT_TYPE_EQ<layout_map<3, 4, 5, 0, 1, 2>, ext_layout_3>();

    typedef typename extend_layout_map<layout, 1, InsertLocation::pre>::type ext_layout_post_1;
    ASSERT_TYPE_EQ<layout_map<0, 1, 2, 3>, ext_layout_post_1>();

    // try the same again with a special layout
    typedef layout_map<2, 1, -1, 0> special_layout;

    typedef typename extend_layout_map<special_layout, 1>::type ext_special_layout_1;
    ASSERT_TYPE_EQ<layout_map<3, 2, -1, 1, 0>, ext_special_layout_1>();

    typedef typename extend_layout_map<special_layout, 2>::type ext_special_layout_2;
    ASSERT_TYPE_EQ<layout_map<4, 3, -1, 2, 0, 1>, ext_special_layout_2>();

    typedef typename extend_layout_map<special_layout, 3>::type ext_special_layout_3;
    ASSERT_TYPE_EQ<layout_map<5, 4, -1, 3, 0, 1, 2>, ext_special_layout_3>();

    typedef typename extend_layout_map<special_layout, 1, InsertLocation::pre>::type ext_special_layout_post_1;
    ASSERT_TYPE_EQ<layout_map<0, 3, 2, -1, 1>, ext_special_layout_post_1>();

    typedef typename extend_layout_map<special_layout, 2, InsertLocation::pre>::type ext_special_layout_post_2;
    ASSERT_TYPE_EQ<layout_map<0, 1, 4, 3, -1, 2>, ext_special_layout_post_2>();
}

TEST(LayoutMap, max_value) {
    {
        using layout_map_t = layout_map<0, 1, 2, 3>;

        GT_STATIC_ASSERT(layout_map_t::max() == 3, " ");
    }
    {
        using layout_map_t = layout_map<0, 1, -1, 2>;

        GT_STATIC_ASSERT(layout_map_t::max() == 2, " ");
    }
    {
        using layout_map_t = layout_map<0, 1, -1, -1>;

        GT_STATIC_ASSERT(layout_map_t::max() == 1, " ");
    }
}
