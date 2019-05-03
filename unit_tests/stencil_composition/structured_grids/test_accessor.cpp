/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/stencil_composition/structured_grids/accessor.hpp>
#include <gridtools/stencil_composition/structured_grids/accessor_mixed.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/stencil_composition/expressions/expressions.hpp>

using namespace gridtools;
using namespace expressions;

static_assert(is_accessor<accessor<6, intent::inout, extent<3, 4, 4, 5>>>::value, "");
static_assert(is_accessor<accessor<2, intent::in>>::value, "");
static_assert(!is_accessor<int>::value, "");
static_assert(!is_accessor<double &>::value, "");
static_assert(!is_accessor<double const &>::value, "");

TEST(accessor, smoke) {
    using testee_t = accessor<0, intent::inout, extent<0, 3, 0, 2, -1, 0>>;
    static_assert(tuple_util::size<testee_t>::value == 3, "");

    testee_t testee{3, 2, -1};

    EXPECT_EQ(3, tuple_util::get<0>(testee));
    EXPECT_EQ(2, tuple_util::get<1>(testee));
    EXPECT_EQ(-1, tuple_util::get<2>(testee));
}

TEST(accessor, zero_accessor) {
    using testee_t = accessor<0>;
    static_assert(tuple_util::size<testee_t>::value == 0, "");
    EXPECT_NO_THROW((testee_t{0, 0, 0, 0}));
    EXPECT_NO_THROW(testee_t{dimension<3>{}});

#ifndef NDEBUG
    EXPECT_THROW(testee_t{1}, std::runtime_error);
    EXPECT_THROW((testee_t{0, 0, 1, 0, 0, 0}), std::runtime_error);
    EXPECT_THROW(testee_t{dimension<3>{4}}, std::runtime_error);
#else
    EXPECT_NO_THROW(testee_t{1});
    EXPECT_NO_THROW((testee_t{0, 0, 1, 0, 0, 0}));
    EXPECT_NO_THROW(testee_t{dimension<3>{4}});
#endif
}

TEST(accessor, extra_args) {
    using testee_t = accessor<0, intent::inout, extent<-1, 1>>;
    static_assert(tuple_util::size<testee_t>::value == 1, "");
    EXPECT_NO_THROW((testee_t{1, 0}));
    EXPECT_NO_THROW(testee_t{dimension<2>{0}});

#ifndef NDEBUG
    EXPECT_THROW((testee_t{0, 1}), std::runtime_error);
    EXPECT_THROW(testee_t{dimension<2>{1}}, std::runtime_error);
#else
    EXPECT_NO_THROW((testee_t{0, 1}));
    EXPECT_NO_THROW(testee_t{dimension<2>{1}});
#endif
}

TEST(accessor, array) {
    accessor<0, intent::inout, extent<0, 3, 0, 2, -1, 0>> first(array<int_t, 3>{3, 2, -1});

    EXPECT_EQ(3, tuple_util::get<0>(first));
    EXPECT_EQ(2, tuple_util::get<1>(first));
    EXPECT_EQ(-1, tuple_util::get<2>(first));
}

/**
 * @brief interface with out-of-order optional arguments
 */
TEST(accessor, alternative1) {
    accessor<0, intent::inout, extent<0, 0, 0, 0>, 6> first(dimension<6>(-6), dimension<4>(12));

    EXPECT_EQ(0, tuple_util::get<0>(first));
    EXPECT_EQ(0, tuple_util::get<1>(first));
    EXPECT_EQ(0, tuple_util::get<2>(first));
    EXPECT_EQ(12, tuple_util::get<3>(first));
    EXPECT_EQ(0, tuple_util::get<4>(first));
    EXPECT_EQ(-6, tuple_util::get<5>(first));
}

/**
 * @brief interface with out-of-order optional arguments, represented as matlab indices
 */
TEST(accessor, alternative2) {
    constexpr dimension<1> i;
    constexpr dimension<2> j;

    constexpr dimension<4> t;
    accessor<0, intent::inout, extent<-5, 0, 0, 0, 0, 8>, 4> first(i - 5, j, dimension<3>(8), t + 2);

    EXPECT_EQ(-5, tuple_util::get<0>(first));
    EXPECT_EQ(0, tuple_util::get<1>(first));
    EXPECT_EQ(8, tuple_util::get<2>(first));
    EXPECT_EQ(2, tuple_util::get<3>(first));
}

/**
 * @brief interface with aliases defined at compile-time
 * allows to split a single field in its different components, assigning an offset to each component.
 * The aforementioned offset is guaranteed to be treated as compile-time static constant value.
 */
TEST(accessor, static_alias) {
    // mixing compile time and runtime values
    using t = dimension<15>;
    using arg_t = accessor<0, intent::inout, extent<0, 4, 0, 0, -5>, 15>;
    using alias_t = alias<arg_t, t, dimension<1>, dimension<7>>::set<-3, 4, 2>;

    alias_t first(dimension<8>(23), dimension<3>(-5));

    EXPECT_EQ(4, tuple_util::get<0>(first));
    EXPECT_EQ(-5, tuple_util::get<2>(first));
    EXPECT_EQ(2, tuple_util::get<6>(first));
    EXPECT_EQ(23, tuple_util::get<7>(first));
    EXPECT_EQ(-3, tuple_util::get<14>(first));
}
