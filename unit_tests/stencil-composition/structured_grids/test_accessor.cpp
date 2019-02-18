/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/stencil-composition/structured_grids/accessor.hpp>
#include <gridtools/stencil-composition/structured_grids/accessor_mixed.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil-composition/accessor_metafunctions.hpp>
#include <gridtools/stencil-composition/expressions/expressions.hpp>
#include <gridtools/stencil-composition/global_accessor.hpp>

using namespace gridtools;
using namespace expressions;

TEST(accessor, is_accessor) {
    GT_STATIC_ASSERT((is_accessor<accessor<6, intent::inout, extent<3, 4, 4, 5>>>::value) == true, "");
    GT_STATIC_ASSERT((is_accessor<accessor<2, intent::in>>::value) == true, "");
    GT_STATIC_ASSERT((is_accessor<int>::value) == false, "");
    GT_STATIC_ASSERT((is_accessor<double &>::value) == false, "");
    GT_STATIC_ASSERT((is_accessor<double const &>::value) == false, "");
}

TEST(accessor, is_accessor_readonly) {
    GT_STATIC_ASSERT((is_accessor_readonly<in_accessor<0>>::value), "");
    GT_STATIC_ASSERT((is_accessor_readonly<accessor<0, intent::in>>::value), "");
    GT_STATIC_ASSERT((is_accessor_readonly<global_accessor<0>>::value), "");
    GT_STATIC_ASSERT((!is_accessor_readonly<inout_accessor<0>>::value), "");
    GT_STATIC_ASSERT((!is_accessor_readonly<accessor<0, intent::inout>>::value), "");
    // TODO test accessor_mixed
}

TEST(accessor, trivial) {
    accessor<0, intent::inout, extent<0, 0, 0, 0>, 3> first(3, 2, -1);

    EXPECT_EQ(3, get<0>(first));
    EXPECT_EQ(2, get<1>(first));
    EXPECT_EQ(-1, get<2>(first));
}

TEST(accessor, array) {
    constexpr accessor<0, intent::inout, extent<0, 0, 0, 0>, 3> first(array<int_t, 3>{3, 2, -1});
    GT_STATIC_ASSERT((get<0>(first) == 3 && get<1>(first) == 2 && get<2>(first) == -1), "ERROR");

    EXPECT_EQ(3, get<0>(first));
    EXPECT_EQ(2, get<1>(first));
    EXPECT_EQ(-1, get<2>(first));
}

/**
 * @brief interface with out-of-order optional arguments
 */
TEST(accessor, alternative1) {
    accessor<0, intent::inout, extent<0, 0, 0, 0>, 6> first(dimension<6>(-6), dimension<4>(12));

    EXPECT_EQ(0, get<0>(first));
    EXPECT_EQ(0, get<1>(first));
    EXPECT_EQ(0, get<2>(first));
    EXPECT_EQ(12, get<3>(first));
    EXPECT_EQ(0, get<4>(first));
    EXPECT_EQ(-6, get<5>(first));
}

/**
 * @brief interface with out-of-order optional arguments, represented as matlab indices
 */
TEST(accessor, alternative2) {
    constexpr dimension<1> i;
    constexpr dimension<2> j;
    constexpr dimension<3> k;

    constexpr dimension<4> t;
#if !defined(__INTEL_COMPILER) || __INTEL_COMPILER != 1800
    // ICC 18 shows some strange bug here
    constexpr accessor<0, intent::inout, extent<0, 0, 0, 0>, 4> first(i - 5, j, dimension<3>(8), t + 2);
    GT_STATIC_ASSERT(get<0>(first) == -5, "ERROR");

    EXPECT_EQ(-5, get<0>(first));
    EXPECT_EQ(0, get<1>(first));
    EXPECT_EQ(8, get<2>(first));
    EXPECT_EQ(2, get<3>(first));
#endif
}

/**
 * @brief interface with aliases defined at compile-time
 * allows to split a single field in its different components, assigning an offset to each component.
 * The aforementioned offset is guaranteed to be treated as compile-time static constant value.
 */
TEST(accessor, static_alias) {
    // mixing compile time and runtime values
    using t = dimension<15>;
    typedef accessor<0, intent::inout, extent<0, 0, 0, 0>, 15> arg_t;
    using alias_t = alias<arg_t, t, dimension<1>, dimension<7>>::set<-3, 4, 2>;

    alias_t first(dimension<8>(23), dimension<3>(-5));

    EXPECT_EQ(2, get<6>(first));
    EXPECT_EQ(4, get<0>(first));
    EXPECT_EQ(-3, get<14>(first));
    EXPECT_EQ(23, get<7>(first));
    EXPECT_EQ(-5, get<2>(first));
}
