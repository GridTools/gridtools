/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#include <boost/variant/apply_visitor.hpp>
#include <gtest/gtest.h>

#include "exported_repository.hpp"
#include <gridtools/interface/repository/repository.hpp>
#include <gridtools/storage/storage-facility.hpp>

#define MY_FIELDTYPES (IJKDataStore)(IJDataStore)
#define MY_FIELDS (IJKDataStore, u)(IJKDataStore, v)(IJDataStore, crlat)
GRIDTOOLS_MAKE_REPOSITORY(my_repository, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS

class simple_repository : public ::testing::Test {
  public:
    my_repository repo;
    simple_repository() : repo(IJKStorageInfo(10, 20, 30), IJStorageInfo(11, 22, 33)) {}
};

TEST_F(simple_repository, access_fields) {
    ASSERT_EQ("u", repo.u().name());
    ASSERT_EQ(10, repo.u().total_length<0>());
    ASSERT_EQ(11, repo.crlat().total_length<0>());
}

TEST_F(simple_repository, assign_to_auto_from_map) {
    // needs a cast
    auto u = boost::get<IJKDataStore>(repo.data_stores()["u"]);
    ASSERT_EQ(10, u.total_length<0>());
}

#ifdef GRIDTOOLS_REPOSITORY_HAS_VARIANT_WITH_IMPLICIT_CONVERSION
TEST_F(simple_repository, assign_to_type_from_map) {
    // no cast needed
    IJKDataStore u = repo.data_stores()["u"];
    ASSERT_EQ(10, u.total_length<0>());
}
#endif

TEST_F(simple_repository, access_wrong_type_from_map) {
    ASSERT_THROW(boost::get<IJDataStore>(repo.data_stores()["u"]), boost::bad_get);
}

class DemonstrateIterationOverMap : public boost::static_visitor<> {
  public:
    std::vector<std::string> names = {"u", "v", "crlat"};
    template <typename T>
    void operator()(T &t) const {
        ASSERT_TRUE(std::find(std::begin(names), std::end(names), t.name()) != std::end(names));
    }
};

TEST_F(simple_repository, iterate_map_with_visitor) {
    for (auto &elem : repo.data_stores()) {
        // can iterate without manual cast
        boost::apply_visitor(DemonstrateIterationOverMap(), elem.second);
    }
}

#define MY_FIELDTYPES (IJKDataStore)
#define MY_FIELDS (IJKDataStore, u)(IJKDataStore, v)
GRIDTOOLS_MAKE_REPOSITORY(my_repository2, MY_FIELDTYPES, MY_FIELDS)
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

    ASSERT_EQ(10, repo.u().total_length<0>());
}

using IJKWStorageInfo = typename gridtools::storage_traits<gridtools::target::x86>::storage_info_t<2, 3>;
using IJKWDataStore =
    typename gridtools::storage_traits<gridtools::target::x86>::data_store_t<gridtools::float_type, IJKWStorageInfo>;
using IKStorageInfo =
    typename gridtools::storage_traits<gridtools::target::x86>::special_storage_info_t<2, gridtools::selector<1, 0, 1>>;
using IKDataStore =
    typename gridtools::storage_traits<gridtools::target::x86>::data_store_t<gridtools::float_type, IKStorageInfo>;

#define MY_FIELDTYPES \
    (IJKDataStore, (0, 1, 2))(IJDataStore, (0, 1, 2))(IJKWDataStore, (0, 1, 3))(IKDataStore, (0, 0, 2))
#define MY_FIELDS (IJKDataStore, u)(IJDataStore, crlat)(IJKWDataStore, w)(IKDataStore, ikfield)
GRIDTOOLS_MAKE_REPOSITORY(my_repository3, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS

TEST(repository_with_dims, constructor) {
    int Ni = 12;
    int Nj = 13;
    int Nk = 14;
    int Nk_plus1 = Nk + 1;

    my_repository3 repo(Ni, Nj, Nk, Nk_plus1);

    ASSERT_EQ(Ni, repo.u().total_length<0>());
    ASSERT_EQ(Nj, repo.u().total_length<1>());
    ASSERT_EQ(Nk, repo.u().total_length<2>());

    ASSERT_EQ(Ni, repo.w().total_length<0>());
    ASSERT_EQ(Nj, repo.w().total_length<1>());
    ASSERT_EQ(Nk_plus1, repo.w().total_length<2>());

    ASSERT_EQ(Ni, repo.crlat().total_length<0>());
    ASSERT_EQ(Nj, repo.crlat().total_length<1>());

    ASSERT_EQ(Ni, repo.ikfield().total_length<0>());
    ASSERT_EQ(Nk, repo.ikfield().total_length<2>());
}

#undef GTREPO_GETTER_PREFIX
#define GTREPO_GETTER_PREFIX get_
#define MY_FIELDTYPES (IJKDataStore)
#define MY_FIELDS (IJKDataStore, u)(IJKDataStore, v)
GRIDTOOLS_MAKE_REPOSITORY(my_repository4, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS

TEST(repository_with_custom_getter_prefix, constructor) {
    int Ni = 12;
    int Nj = 13;
    int Nk = 14;

    my_repository4 repo(IJKStorageInfo(Ni, Nj, Nk));

    ASSERT_EQ(Ni, repo.get_u().total_length<0>());
    ASSERT_EQ(Nj, repo.get_u().total_length<1>());
    ASSERT_EQ(Nk, repo.get_u().total_length<2>());

    ASSERT_EQ(Ni, repo.get_v().total_length<0>());
    ASSERT_EQ(Nj, repo.get_v().total_length<1>());
    ASSERT_EQ(Nk, repo.get_v().total_length<2>());
}

extern "C" void call_repository(); // implemented in test_repository.f90
TEST(repository_with_custom_getter_prefix, fortran_bindings) {
    // the test for this code is in exported_repository.cpp
    call_repository();
}
