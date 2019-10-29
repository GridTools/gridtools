/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "exported_repository.hpp"
#include <boost/variant/apply_visitor.hpp>
#include <gmock/gmock.h>
#include <gridtools/interface/repository/repository.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gtest/gtest.h>

using testing::ElementsAre;

#define MY_FIELDTYPES (IJKDataStore)(IJDataStore)
#define MY_FIELDS (IJKDataStore, u)(IJKDataStore, v)(IJDataStore, crlat)
GT_MAKE_REPOSITORY(my_repository, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS

class simple_repository : public ::testing::Test {
  public:
    my_repository repo;
    simple_repository() : repo(IJKStorageInfo(10, 20, 30), IJStorageInfo(11, 22, 33)) {}
};

TEST_F(simple_repository, access_fields) {
    EXPECT_EQ("u", repo.u().name());
    EXPECT_EQ(10, repo.u().lengths()[0]);
    EXPECT_EQ(11, repo.crlat().lengths()[0]);
}

TEST_F(simple_repository, assign_to_auto_from_map) {
    // needs a cast
    auto &&u = boost::get<IJKDataStore>(repo.data_stores().at("u"));
    EXPECT_EQ(10, u.lengths()[0]);
}

#ifdef GT_REPOSITORY_HAS_VARIANT_WITH_IMPLICIT_CONVERSION
TEST_F(simple_repository, assign_to_type_from_map) {
    // no cast needed
    IJKDataStore u = repo.data_stores().at("u");
    EXPECT_EQ(10, u.lengths()[0]);
}
#endif

TEST_F(simple_repository, access_wrong_type_from_map) {
    ASSERT_THROW(boost::get<IJDataStore>(repo.data_stores().at("u")), boost::bad_get);
}

TEST_F(simple_repository, iterate_map_with_visitor) {
    for (auto &&elem : repo.data_stores())
        boost::apply_visitor([&](auto &&x) { EXPECT_EQ(elem.first, x.name()); }, elem.second);
}

#define MY_FIELDTYPES (IJKDataStore)
#define MY_FIELDS (IJKDataStore, u)(IJKDataStore, v)
GT_MAKE_REPOSITORY(my_repository2, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS

TEST(two_repository, test) {
    my_repository repo1(IJKStorageInfo(10, 20, 30), IJStorageInfo(11, 22, 33));
    my_repository2 repo2(IJKStorageInfo(22, 33, 44));

    ASSERT_EQ(3, repo1.data_stores().size());
    ASSERT_EQ(2, repo2.data_stores().size());
}

class my_extended_repo : public my_repository {
    using my_repository::my_repository;
    void some_extra_function() {}
};

TEST(extended_repo, inherited_functions) {
    my_extended_repo repo(IJKStorageInfo(10, 20, 30), IJStorageInfo(11, 22, 33));

    EXPECT_EQ(10, repo.u().lengths()[0]);
}

using IJKWStorageInfo = typename gridtools::storage_traits<gridtools::backend::x86>::storage_info_t<2, 3>;
using IJKWDataStore =
    typename gridtools::storage_traits<gridtools::backend::x86>::data_store_t<float_type, IJKWStorageInfo>;
using IKStorageInfo = typename gridtools::storage_traits<gridtools::backend::x86>::special_storage_info_t<2,
    gridtools::selector<1, 0, 1>>;
using IKDataStore =
    typename gridtools::storage_traits<gridtools::backend::x86>::data_store_t<float_type, IKStorageInfo>;

#define MY_FIELDTYPES \
    (IJKDataStore, (0, 1, 2))(IJDataStore, (0, 1, 2))(IJKWDataStore, (0, 1, 3))(IKDataStore, (0, 0, 2))
#define MY_FIELDS (IJKDataStore, u)(IJDataStore, crlat)(IJKWDataStore, w)(IKDataStore, ikfield)
GT_MAKE_REPOSITORY(my_repository3, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS

TEST(repository_with_dims, constructor) {
    int Ni = 12;
    int Nj = 13;
    int Nk = 14;
    int Nk_plus1 = Nk + 1;

    my_repository3 repo(Ni, Nj, Nk, Nk_plus1);

    EXPECT_THAT(repo.u().lengths(), ElementsAre(Ni, Nj, Nk));
    EXPECT_THAT(repo.w().lengths(), ElementsAre(Ni, Nj, Nk_plus1));
    EXPECT_THAT(repo.crlat().lengths(), ElementsAre(Ni, Nj, Nk));
    EXPECT_THAT(repo.ikfield().lengths(), ElementsAre(Ni, Ni, Nk));
}

#undef GT_REPO_GETTER_PREFIX
#define GT_REPO_GETTER_PREFIX get_
#define MY_FIELDTYPES (IJKDataStore)
#define MY_FIELDS (IJKDataStore, u)(IJKDataStore, v)
GT_MAKE_REPOSITORY(my_repository4, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS

TEST(repository_with_custom_getter_prefix, constructor) {
    int Ni = 12;
    int Nj = 13;
    int Nk = 14;

    my_repository4 repo(IJKStorageInfo(Ni, Nj, Nk));

    EXPECT_THAT(repo.get_u().lengths(), ElementsAre(Ni, Nj, Nk));
    EXPECT_THAT(repo.get_v().lengths(), ElementsAre(Ni, Nj, Nk));
}
