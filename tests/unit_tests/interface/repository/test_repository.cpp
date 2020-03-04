/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/interface/repository.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/x86.hpp>

const auto builder = gridtools::storage::builder<gridtools::storage::x86>.type<double>();
auto ijk_builder(int i, int j) { return builder.dimensions(i, j, 80); }
auto ij_builder(int i, int j) { return builder.selector<1, 1, 0>().dimensions(i + 2, j + 2, 80).halos(1, 1, 0); }

GT_DEFINE_REPOSITORY(my_repository, (ijk, ijk_builder)(ij, ij_builder), (ijk, u)(ijk, v)(ij, crlat));

TEST(test_repository, access_fields) {
    my_repository repo(10, 20);
    EXPECT_EQ("u", repo.u->name());
    EXPECT_EQ(10, repo.u->lengths()[0]);
    EXPECT_EQ(12, repo.crlat->lengths()[0]);
}

TEST(test_repository, map) {
    my_repository repo(10, 20);
    EXPECT_EQ("u", repo.ijk("u")->name());
    EXPECT_EQ("crlat", repo.ij("crlat")->name());
}

TEST(test_repository, wrong_type_from_map) {
    my_repository repo(10, 20);
    ASSERT_THROW(repo.ijk("crlat"), std::runtime_error);
    ASSERT_THROW(repo.ijk("junk"), std::runtime_error);
}

TEST(test_repository, iterate_map_with_visitor) {
    my_repository repo(10, 20);
    std::vector<std::string> names;
    repo.for_each([&](auto ds) { names.push_back(ds->name()); });
    EXPECT_THAT(names, testing::ElementsAre("u", "v", "crlat"));
}
