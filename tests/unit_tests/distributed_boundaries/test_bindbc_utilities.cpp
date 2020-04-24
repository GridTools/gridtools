/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/distributed_boundaries/bound_bc.hpp>

#include <utility>

#include <gtest/gtest.h>

#include <gridtools/boundaries/zero.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/x86.hpp>

using namespace std::placeholders;
namespace gt = gridtools;

TEST(DistributedBoundaries, SelectElement) {
    auto all = std::make_tuple(1, _1, 3, _2);
    auto sub = std::make_tuple(2, 4);

    EXPECT_EQ(gt::_impl::select_element<0>(sub, all, gt::_impl::NotPlc{}), 1);
    EXPECT_EQ(gt::_impl::select_element<1>(sub, all, gt::_impl::Plc{}), 2);
    EXPECT_EQ(gt::_impl::select_element<2>(sub, all, gt::_impl::NotPlc{}), 3);
    EXPECT_EQ(gt::_impl::select_element<3>(sub, all, gt::_impl::Plc{}), 4);
}

namespace collect_indices {
    template <class Tuple, size_t... Is>
    constexpr bool testee = std::is_same<
        typename gt::_impl::comm_indices<std::tuple<>>::collect_indices<0, std::index_sequence<>, Tuple>::type,
        std::index_sequence<Is...>>::value;

    static_assert(testee<std::tuple<int, int>, 0, 1>, "");
    static_assert(testee<std::tuple<int, decltype(_1), int, decltype(_2)>, 0, 2>, "");
    static_assert(testee<std::tuple<decltype(_1), decltype(_2)>>, "");
} // namespace collect_indices

TEST(DistributedBoundaries, RestTuple) {
    EXPECT_EQ(gt::_impl::rest_tuple(std::make_tuple(), std::make_index_sequence<0>{}), std::make_tuple());
    EXPECT_EQ(gt::_impl::rest_tuple(std::make_tuple(1), std::make_index_sequence<0>{}), std::make_tuple());
    EXPECT_EQ(gt::_impl::rest_tuple(std::make_tuple(1, 2), std::make_index_sequence<1>{}), std::make_tuple(2));
}

static_assert(!gt::_impl::contains_placeholders<decltype(std::make_tuple(3, 4, 5))>::value, "");
static_assert(!gt::_impl::contains_placeholders<decltype(std::make_tuple())>::value, "");
static_assert(gt::_impl::contains_placeholders<decltype(std::make_tuple(3, 4, _1))>::value, "");
static_assert(gt::_impl::contains_placeholders<decltype(std::make_tuple(3, _2, 5))>::value, "");

TEST(DistributedBoundaries, BoundBC) {
    const auto builder = gt::storage::builder<gt::storage::x86>.type<double>().dimensions(3, 3, 3);

    auto a = builder();
    auto b = builder();
    auto c = builder();

    using ds = decltype(a);

    gt::bound_bc<gt::zero_boundary, std::tuple<ds, ds, ds>, std::index_sequence<1>> bbc{
        gt::zero_boundary{}, std::make_tuple(a, b, c)};

    auto x = bbc.stores();

    EXPECT_EQ(a, std::get<0>(x));
    EXPECT_EQ(b, std::get<1>(x));
    EXPECT_EQ(c, std::get<2>(x));

    auto y = bbc.exc_stores();

    EXPECT_EQ(std::tuple_size<decltype(y)>::value, 1);

    EXPECT_EQ(b, std::get<0>(y));
}
