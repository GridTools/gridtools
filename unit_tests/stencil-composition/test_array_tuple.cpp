/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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

#include "gtest/gtest.h"
#include <gridtools.hpp>
#include <stencil-composition/stencil-composition.hpp>

namespace array_tuple_test {
    using namespace gridtools;

    template < int_t... S >
    struct storage_stub {

        static const int_t space_dimensions = sizeof...(S);
        array< int_t, space_dimensions > m_strides;

        storage_stub() : m_strides{S...} {}
        int_t const &strides(int_t id) { return m_strides[id]; }
    };

    template < typename T1, typename T2 >
    struct pair {
        typedef T1 first;
        typedef T2 second;
    };

    bool test() {

        typedef
            typename boost::mpl::vector< storage_stub< 0, 1, 2 >, storage_stub< 3, 4 >, storage_stub< 5, 6, 7, 8, 9 > >
                storage_vec_t;
        array_tuple< 2, storage_vec_t, int_t, 1 > at;

        storage_stub< 0, 1, 2 > *s1 = new storage_stub< 0, 1, 2 >();
        storage_stub< 3, 4 > *s2 = new storage_stub< 3, 4 >;
        storage_stub< 5, 6, 7, 8, 9 > *s3 = new storage_stub< 5, 6, 7, 8, 9 >();
        auto svec = boost::fusion::make_vector(s1, s2, s3);

        assign_strides_functor< array_tuple< 2, storage_vec_t, int_t, 1 >, decltype(svec) > func(at, svec);
        func(pair< storage_stub< 0, 1, 2 >, static_int< 0 > >());
        func(pair< storage_stub< 3, 4 >, static_int< 1 > >());
        func(pair< storage_stub< 5, 6, 7, 8, 9 >, static_int< 2 > >());

        bool success = true;
        success = success && (at.template get< 0 >()[0] == 1);
        success = success && (at.template get< 0 >()[1] == 2);
        success = success && (at.template get< 1 >()[0] == 4);
        success = success && (at.template get< 2 >()[0] == 6);
        success = success && (at.template get< 2 >()[1] == 7);
        success = success && (at.template get< 2 >()[2] == 8);
        success = success && (at.template get< 2 >()[3] == 9);
        return success;
    }

} // namespace array_tuple_test

TEST(array_tuple_test, functionality_test) { EXPECT_EQ(array_tuple_test::test(), true); }
