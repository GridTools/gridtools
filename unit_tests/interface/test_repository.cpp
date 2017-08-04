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

#include <gtest/gtest.h>
#include <boost/variant/apply_visitor.hpp>

#include "storage/storage-facility.hpp"
#include "interface/repository/repository.hpp"

using namespace gridtools;

using IJKStorageInfo = typename storage_traits< enumtype::Host >::storage_info_t< 0, 3 >;
using IJKDataStore = typename storage_traits< enumtype::Host >::data_store_t< float_type, IJKStorageInfo >;
using IJStorageInfo = typename storage_traits< enumtype::Host >::storage_info_t< 1, 2 >;
using IJDataStore = typename storage_traits< enumtype::Host >::data_store_t< float_type, IJStorageInfo >;

#define MY_FIELDTYPES (IJKDataStore)(IJDataStore)
#define MY_FIELDS (IJKDataStore, u)(IJKDataStore, v)(IJDataStore, crlat)
GT_MAKE_REPOSITORY(my_repository, MY_FIELDTYPES, MY_FIELDS)
#undef MY_FIELDTYPES
#undef MY_FIELDS

class simple_repository : public ::testing::Test {
  public:
    my_repository repo;
    simple_repository() : repo(IJKStorageInfo(10, 20, 30), IJStorageInfo(11, 22)) {}
};

TEST_F(simple_repository, access_fields) {
    ASSERT_EQ("u", repo.get_u().name());
    ASSERT_EQ(10, repo.get_u().dim< 0 >());
    ASSERT_EQ(11, repo.get_crlat().dim< 0 >());
}

TEST_F(simple_repository, access_field_from_map) {
    // needs a cast
    auto u = boost::get< IJKDataStore >(repo.data_stores()["u"]);

    ASSERT_EQ(10, u.dim< 0 >());
}

class DemonstrateIterationOverMap : public boost::static_visitor<> {
  public:
    std::vector< std::string > names = {"u", "v", "crlat"};
    template < typename T >
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
GT_MAKE_REPOSITORY(my_repository2, MY_FIELDTYPES, MY_FIELDS)

TEST(two_repository, test) {
    my_repository repo1(IJKStorageInfo(10, 20, 30), IJStorageInfo(11, 22));
    my_repository2 repo2(IJKStorageInfo(22, 33, 44));

    ASSERT_EQ(3, repo1.data_stores().size());
    ASSERT_EQ(2, repo2.data_stores().size());
}

class my_extended_repo : public my_repository {
    using my_repository::my_repository;
    void some_extra_function() {}
};

TEST(extended_repo, inherited_functions) {
    my_extended_repo repo(IJKStorageInfo(10, 20, 30), IJStorageInfo(11, 22));

    ASSERT_EQ(10, repo.get_u().dim< 0 >());
}

// TODO notes form discussion with Carlos
// int dims[3] {...};
// wrap = init_wrappable( "dycore", dims );
//
//
// class wrappable : repository
//{}
//
// class dycore: input_output_repo:wrappable
//{
//	dycore( std::vector<int> dims ): wrappable( storage_info1(dim[0],dim[1],dim[2]),
// storage_info2(dim[0],dim[1],dim[2]+1))
//	{
//
//	}
//};
//
// MAKE_REPO(const)
//
////class const_repository;
////class const_wrappable_repo
//
// class wrappable: repository
//
// class const_repository: wrappable
//{
//	constantFields( std::vector<int> dims ): wrappable( storage_info1(dim[0]), storage_info2(3))
//	{
//
//	}
// private:
//
//};
