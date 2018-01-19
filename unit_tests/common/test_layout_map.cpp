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

#include <common/layout_map.hpp>
#include <common/layout_map_metafunctions.hpp>
#include <common/gt_assert.hpp>

#include "../test_helper.hpp"
using namespace gridtools;

template < typename T >
constexpr uint_t get_length() {
    return T::masked_length;
}

TEST(LayoutMap, SimpleLayout) {
    typedef layout_map< 0, 1, 2 > layout1;

    // test length
    GRIDTOOLS_STATIC_ASSERT(layout1::masked_length == 3, "layout_map length is wrong");
    GRIDTOOLS_STATIC_ASSERT(layout1::unmasked_length == 3, "layout_map length is wrong");

    // test find method
    GRIDTOOLS_STATIC_ASSERT(layout1::find< 0 >() == 0, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout1::find< 1 >() == 1, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout1::find< 2 >() == 2, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout1::find(0) == 0, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout1::find(1) == 1, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout1::find(2) == 2, "wrong result in layout_map find method");

    // test at method
    GRIDTOOLS_STATIC_ASSERT(layout1::at< 0 >() == 0, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout1::at< 1 >() == 1, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout1::at< 2 >() == 2, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout1::at(0) == 0, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout1::at(1) == 1, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout1::at(2) == 2, "wrong result in layout_map at method");
}

TEST(LayoutMap, ExtendedLayout) {
    typedef layout_map< 3, 2, 1, 0 > layout2;

    // test length
    GRIDTOOLS_STATIC_ASSERT(layout2::masked_length == 4, "layout_map length is wrong");
    GRIDTOOLS_STATIC_ASSERT(layout2::unmasked_length == 4, "layout_map length is wrong");

    // test find method
    GRIDTOOLS_STATIC_ASSERT(layout2::find< 0 >() == 3, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout2::find< 1 >() == 2, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout2::find< 2 >() == 1, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout2::find< 3 >() == 0, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout2::find(0) == 3, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout2::find(1) == 2, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout2::find(2) == 1, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout2::find(3) == 0, "wrong result in layout_map find method");

    // test at method
    GRIDTOOLS_STATIC_ASSERT(layout2::at< 0 >() == 3, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout2::at< 1 >() == 2, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout2::at< 2 >() == 1, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout2::at< 3 >() == 0, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout2::at(0) == 3, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout2::at(1) == 2, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout2::at(2) == 1, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout2::at(3) == 0, "wrong result in layout_map at method");
}

TEST(LayoutMap, MaskedLayout) {
    typedef layout_map< 2, -1, 1, 0 > layout3;

    // test length
    GRIDTOOLS_STATIC_ASSERT(layout3::masked_length == 4, "layout_map length is wrong");
    GRIDTOOLS_STATIC_ASSERT(layout3::unmasked_length == 3, "layout_map length is wrong");

    // test find method
    GRIDTOOLS_STATIC_ASSERT(layout3::find< 0 >() == 3, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout3::find< 1 >() == 2, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout3::find< 2 >() == 0, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout3::find(0) == 3, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout3::find(1) == 2, "wrong result in layout_map find method");
    GRIDTOOLS_STATIC_ASSERT(layout3::find(2) == 0, "wrong result in layout_map find method");

    // test at method
    GRIDTOOLS_STATIC_ASSERT(layout3::at< 0 >() == 2, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout3::at< 1 >() == -1, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout3::at< 2 >() == 1, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout3::at< 3 >() == 0, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout3::at(0) == 2, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout3::at(1) == -1, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout3::at(2) == 1, "wrong result in layout_map at method");
    GRIDTOOLS_STATIC_ASSERT(layout3::at(3) == 0, "wrong result in layout_map at method");
}

TEST(LayoutMap, DefaultLayout) {
    ASSERT_TYPE_EQ< layout_map< 0 >, default_layout_map< 1 >::type >();
    ASSERT_TYPE_EQ< layout_map< 0, 1 >, default_layout_map< 2 >::type >();
    ASSERT_TYPE_EQ< layout_map< 0, 1, 2 >, default_layout_map< 3 >::type >();
    ASSERT_TYPE_EQ< layout_map< 0, 1, 2, 3 >, default_layout_map< 4 >::type >();

    ASSERT_TYPE_EQ< layout_map< 0, 1, 2, 3 >, default_layout_map_t< 4 > >();
}

TEST(LayoutMap, Extender) {
    typedef layout_map< 0, 1, 2 > layout;
    // create some extended layout maps
    // the extension by 1 dimensions means that all original layout_map args are increased by 1 and a 0 is added at the
    // end
    typedef typename extend_layout_map< layout, 1 >::type ext_layout_1;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< ext_layout_1, layout_map< 1, 2, 3, 0 > >::type::value), "layout_map type is wrong");
    // the extension by 2 dimensions means that all original layout_map args are increased by 2 and 0, 1 is added at the
    // end
    typedef typename extend_layout_map< layout, 2 >::type ext_layout_2;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< ext_layout_2, layout_map< 2, 3, 4, 0, 1 > >::type::value), "layout_map type is wrong");
    // the extension by 3 dimensions means that all original layout_map args are increased by 3 and 0, 1, 2 is added at
    // the end
    typedef typename extend_layout_map< layout, 3 >::type ext_layout_3;
    GRIDTOOLS_STATIC_ASSERT(
        (boost::is_same< ext_layout_3, layout_map< 3, 4, 5, 0, 1, 2 > >::type::value), "layout_map type is wrong");

    // try the same again with a special layout
    typedef layout_map< 2, 1, -1, 0 > special_layout;
    // the extension by 1 dimensions means that all original layout_map args are increased by 1 and a 0 is added at the
    // end
    typedef typename extend_layout_map< special_layout, 1 >::type ext_special_layout_1;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< ext_special_layout_1, layout_map< 3, 2, -1, 1, 0 > >::type::value),
        "layout_map type is wrong");
    // the extension by 2 dimensions means that all original layout_map args are increased by 2 and 0, 1 is added at the
    // end
    typedef typename extend_layout_map< special_layout, 2 >::type ext_special_layout_2;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< ext_special_layout_2, layout_map< 4, 3, -1, 2, 0, 1 > >::type::value),
        "layout_map type is wrong");
    // the extension by 3 dimensions means that all original layout_map args are increased by 3 and 0, 1, 2 is added at
    // the end
    typedef typename extend_layout_map< special_layout, 3 >::type ext_special_layout_3;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< ext_special_layout_3, layout_map< 5, 4, -1, 3, 0, 1, 2 > >::type::value),
        "layout_map type is wrong");
}
