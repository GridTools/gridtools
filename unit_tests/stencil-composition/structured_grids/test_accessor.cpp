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
#include "gtest/gtest.h"

#include <common/defs.hpp>
#include <stencil-composition/structured_grids/accessor.hpp>
#include <stencil-composition/structured_grids/accessor_metafunctions.hpp>
#include <stencil-composition/structured_grids/vector_accessor.hpp>
#include <stencil-composition/global_accessor.hpp>
#include <stencil-composition/expressions/expressions.hpp>

using namespace gridtools;

/** @brief simple interface
 */
bool test_trivial() {
    accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 3 > first(3, 2, -1);
    return first.get< 2 >() == 3 && first.get< 1 >() == 2 && first.get< 0 >() == -1;
}

bool test_array() {
    constexpr accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 3 > first(array< int_t, 3 >{3, 2, -1});
    GRIDTOOLS_STATIC_ASSERT((first.get< 2 >() == 3 && first.get< 1 >() == 2 && first.get< 0 >() == -1), "ERROR");
    return first.get< 2 >() == 3 && first.get< 1 >() == 2 && first.get< 0 >() == -1;
}

/** @brief interface with out-of-order optional arguments
 */
bool test_alternative1() {
    accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 6 > first(dimension< 6 >(-6), dimension< 4 >(12));

    return first.get< 5 - 0 >() == 0 && first.get< 5 - 1 >() == 0 && first.get< 5 - 2 >() == 0 &&
           first.get< 5 - 3 >() == 12 && first.get< 5 - 4 >() == 0 && first.get< 5 - 5 >() == -6;
}

/** @brief interface with out-of-order optional arguments, represented as matlab indices
 */

using namespace expressions;

bool test_alternative2() {

    constexpr dimension< 1 > i;
    constexpr dimension< 2 > j;
    constexpr dimension< 3 > k;

    constexpr dimension< 4 > t;
    constexpr accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 4 > first(i - 5, j, dimension< 3 >(8), t + 2);

    GRIDTOOLS_STATIC_ASSERT(first.get< 3 - 0 >() == -5, "ERROR");
    return first.get< 3 - 0 >() == -5 && first.get< 3 - 1 >() == 0 && first.get< 3 - 2 >() == 8 &&
           first.get< 3 - 3 >() == 2;
}

/** @brief interface with aliases defined at compile-time

    allows to split a single field in its different components, assigning an offset to each component.
    The aforementioned offset is guaranteed to be treated as compile-time static constant value.
*/
bool test_static_alias() {

    // mixing compile time and runtime values
    using t = dimension< 15 >;
    typedef accessor< 0, enumtype::inout, extent< 0, 0, 0, 0 >, 15 > arg_t;
    using alias_t = alias< arg_t, t, dimension< 1 >, dimension< 7 > >::set< -3, 4, 2 >;

    alias_t first(dimension< 8 >(23), dimension< 3 >(-5));

    return first.get< 14 - 6 >() == 2 && first.get< 14 - 0 >() == 4 && first.get< 14 - 14 >() == -3 &&
           first.get< 14 - 7 >() == 23 && first.get< 14 - 2 >() == -5;
}

TEST(accessor, is_accessor) {
    GRIDTOOLS_STATIC_ASSERT((is_accessor< accessor< 6, enumtype::inout, extent< 3, 4, 4, 5 > > >::value) == true, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< accessor< 2, enumtype::in > >::value) == true, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< int >::value) == false, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< double & >::value) == false, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< double const & >::value) == false, "");
}

TEST(accessor, is_accessor_readonly) {
    GRIDTOOLS_STATIC_ASSERT((is_accessor_readonly< in_accessor< 0 > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor_readonly< accessor< 0, enumtype::in > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor_readonly< vector_accessor< 0, enumtype::in > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor_readonly< global_accessor< 0 > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_accessor_readonly< inout_accessor< 0 > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_accessor_readonly< accessor< 0, enumtype::inout > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_accessor_readonly< vector_accessor< 0, enumtype::inout > >::value), "");
    // TODO test accessor_mixed
}

TEST(accessor, is_grid_accessor) {
    GRIDTOOLS_STATIC_ASSERT((is_grid_accessor< accessor< 0, enumtype::in > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((is_grid_accessor< vector_accessor< 0, enumtype::in > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_grid_accessor< global_accessor< 0 > >::value), "");
}

TEST(accessor, is_regular_accessor) {
    GRIDTOOLS_STATIC_ASSERT((is_regular_accessor< accessor< 0, enumtype::in > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_regular_accessor< vector_accessor< 0, enumtype::in > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_regular_accessor< global_accessor< 0 > >::value), "");
}

TEST(accessor, is_vector_accessor) {
    GRIDTOOLS_STATIC_ASSERT((is_vector_accessor< vector_accessor< 0, enumtype::in > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_vector_accessor< accessor< 0, enumtype::in > >::value), "");
    GRIDTOOLS_STATIC_ASSERT((!is_vector_accessor< global_accessor< 0 > >::value), "");
}

TEST(accessor, copy_const) {

    // TODOCOSUNA not working due to problems with the copy ctor of the accessors

    //    typedef accessor<0, extent<-1,0,0,0>, 3> accessor_t;
    //    accessor<0, extent<-1,0,0,0>, 3> in(1,2,3);
    //    accessor<1, extent<-1,0,0,0>, 3> out(in);
    //
    //    ASSERT_TRUE(in.get<0>() == 3 && out.get<0>()==3);
    //    ASSERT_TRUE(in.get<1>() == 2 && out.get<1>()==2);
    //    ASSERT_TRUE(in.get<2>() == 1 && out.get<2>()==1);
    //
    //    typedef boost::mpl::map1<
    //        boost::mpl::pair<
    //            boost::mpl::integral_c<int, 0>, boost::mpl::integral_c<int, 8>
    //        >
    //    > ArgsMap;
    //
    //    typedef remap_accessor_type<accessor_t, ArgsMap>::type remap_accessor_t;
    //
    //    BOOST_STATIC_ASSERT((is_accessor<remap_accessor_t>::value));
    //    BOOST_STATIC_ASSERT((accessor_index<remap_accessor_t>::value == 8));
    //
    //    ASSERT_TRUE(remap_accessor_t(in).get<0>() == 3);
    //    ASSERT_TRUE(remap_accessor_t(in).get<1>() == 2);
    //    ASSERT_TRUE(remap_accessor_t(in).get<2>() == 1);
}

TEST(accessor, trivial) { EXPECT_TRUE(test_trivial()); }

TEST(accessor, array) { EXPECT_TRUE(test_array()); }

TEST(accessor, alternative1) { EXPECT_TRUE(test_alternative1()); }

TEST(accessor, alternative2) { EXPECT_TRUE(test_alternative2()); }

TEST(accessor, static_alias) { EXPECT_TRUE(test_static_alias()); }
