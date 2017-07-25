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
#include "my_repository.hpp"
#include <boost/variant/apply_visitor.hpp>

TEST(repository, init_a_member) {
    my_repository repo(IJKStorageInfo(10, 20, 30), IJStorageInfo(11, 22));

    ASSERT_EQ(10, repo.get_u().dim< 0 >());
    ASSERT_EQ(11, repo.get_crlat().dim< 0 >());
}

// TEST(repository, type_to_id_and_back) {
//    GRIDTOOLS_STATIC_ASSERT((std::is_same< typename id_to_type< static_int< type_to_id< IJKDataStore >::value >
//    >::type,
//                                IJKDataStore >::value),
//        "ERROR");
//}

TEST(test_variant, test1) {
    boost::variant< int, double, float > test(5);
    //    float as_f = boost::get< float >(test);
    int as_int = boost::get< int >(test);
    std::cout << test << std::endl;
}

TEST(repository, access_map) {
    my_repository repo(IJKStorageInfo(10, 20, 30), IJStorageInfo(11, 22));

    auto u = boost::get< IJKDataStore >(repo.data_store_map_["u"]);

    ASSERT_EQ(10, u.dim< 0 >());
}

class PrintDim : public boost::static_visitor<> {
  public:
    template < typename T >
    void operator()(T &t) const {
        std::cout << t.template dim< 0 >() << std::endl;
    }
};

TEST(repository, iterate) {
    my_repository repo(IJKStorageInfo(10, 20, 30), IJStorageInfo(11, 22));

    for (auto &elem : repo.data_store_map_) {
        boost::apply_visitor(PrintDim(), elem.second);
    }
}

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
