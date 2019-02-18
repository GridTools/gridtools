/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "gtest/gtest.h"
#include <gridtools/boundary-conditions/zero.hpp>
#include <gridtools/distributed-boundaries/bound_bc.hpp>
#include <gridtools/storage/data_store.hpp>
#include <gridtools/storage/storage_host/host_storage.hpp>

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

TEST(DistributedBoundaries, DataStoreOrPlc) {
    typedef gt::storage_info_interface<0, gt::layout_map<0, 1, 2>> storage_info_t;
    using ds = gt::data_store<gt::host_storage<double>, storage_info_t>;

    EXPECT_EQ((gt::_impl::data_stores_or_placeholders<decltype(_1), decltype(_2)>()), true);
    EXPECT_EQ((gt::_impl::data_stores_or_placeholders<ds, ds>()), true);
    EXPECT_EQ((gt::_impl::data_stores_or_placeholders<decltype(_1), ds, decltype(_2), ds>()), true);
    EXPECT_EQ((gt::_impl::data_stores_or_placeholders<decltype(_1), int, decltype(_2)>()), false);
    EXPECT_EQ((gt::_impl::data_stores_or_placeholders<ds, int, ds>()), false);
    EXPECT_EQ((gt::_impl::data_stores_or_placeholders<decltype(_1), ds, int, decltype(_2), ds>()), false);
}

TEST(DistributedBoundaries, CollectIndices) {
    using p1 = decltype(_1);
    using p2 = decltype(_2);

    EXPECT_TRUE(
        (std::is_same<typename gt::_impl::comm_indices<
                          std::tuple<>>::collect_indices<0, gt::meta::index_sequence<>, std::tuple<int, int>>::type,
            gt::meta::index_sequence<0, 1>>::value));

    EXPECT_TRUE((std::is_same<typename gt::_impl::comm_indices<std::tuple<>>::
                                  collect_indices<0, gt::meta::index_sequence<>, std::tuple<int, p1, int, p2>>::type,
        gt::meta::index_sequence<0, 2>>::value));

    EXPECT_TRUE(
        (std::is_same<typename gt::_impl::comm_indices<
                          std::tuple<>>::collect_indices<0, gt::meta::index_sequence<>, std::tuple<p1, p2>>::type,
            gt::meta::index_sequence<>>::value));
}

TEST(DistributedBoundaries, RestTuple) {
    {
        auto all = std::make_tuple();
        EXPECT_EQ(gt::_impl::rest_tuple(all, gt::meta::make_index_sequence<0>{}), (std::tuple<>{}));
    }
    {
        auto all = std::make_tuple(1);
        EXPECT_EQ(gt::_impl::rest_tuple(all, gt::meta::make_index_sequence<0>{}), (std::tuple<>{}));
    }
    {
        auto all = std::make_tuple(1, 2);
        EXPECT_EQ(gt::_impl::rest_tuple(all, gt::meta::make_index_sequence<1>{}), (std::tuple<int>{2}));
    }
}

TEST(DistributedBoundaries, ContainsPlaceholders) {
    {
        auto x = std::make_tuple(3, 4, 5);
        EXPECT_FALSE(gt::_impl::contains_placeholders<decltype(x)>::value);
    }

    {
        auto x = std::make_tuple();
        EXPECT_FALSE(gt::_impl::contains_placeholders<decltype(x)>::value);
    }

    {
        auto x = std::make_tuple(3, 4, _1);
        EXPECT_TRUE(gt::_impl::contains_placeholders<decltype(x)>::value);
    }

    {
        auto x = std::make_tuple(3, _2, 5);
        EXPECT_TRUE(gt::_impl::contains_placeholders<decltype(x)>::value);
    }
}

TEST(DistributedBoundaries, BoundBC) {
    typedef gt::storage_info_interface<0, gt::layout_map<0, 1, 2>> storage_info_t;
    using ds = gt::data_store<gt::host_storage<double>, storage_info_t>;

    ds a(storage_info_t{3, 3, 3}, "a");
    ds b(storage_info_t{3, 3, 3}, "b");
    ds c(storage_info_t{3, 3, 3}, "c");

    gt::bound_bc<gt::zero_boundary, std::tuple<ds, ds, ds>, gt::meta::index_sequence<1>> bbc{
        gt::zero_boundary{}, std::make_tuple(a, b, c)};

    auto x = bbc.stores();

    EXPECT_EQ(a, std::get<0>(x));
    EXPECT_EQ(b, std::get<1>(x));
    EXPECT_EQ(c, std::get<2>(x));

    auto y = bbc.exc_stores();

    EXPECT_EQ(std::tuple_size<decltype(y)>::value, 1);

    EXPECT_EQ(b, std::get<0>(y));
}
