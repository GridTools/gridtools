/*
  GridTools Libraries

  Copyright (c) 2017, GridTools Consortium
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

#include "common/layout_map.hpp"

using namespace gridtools;

template <typename T>
constexpr unsigned get_length() {
    return T::masked_length;
}

TEST(LayoutMap, SimpleLayout) {
    typedef layout_map<0,1,2> layout1;

    // test length
    static_assert(layout1::masked_length==3, "layout_map length is wrong");
    static_assert(layout1::unmasked_length==3, "layout_map length is wrong");
    
    // test find method
    static_assert(layout1::find<0>()==0, "wrong result in layout_map find method");
    static_assert(layout1::find<1>()==1, "wrong result in layout_map find method");
    static_assert(layout1::find<2>()==2, "wrong result in layout_map find method");
    static_assert(layout1::find(0)==0, "wrong result in layout_map find method");
    static_assert(layout1::find(1)==1, "wrong result in layout_map find method");
    static_assert(layout1::find(2)==2, "wrong result in layout_map find method");
    
    // test at method
    static_assert(layout1::at<0>()==0, "wrong result in layout_map at method");
    static_assert(layout1::at<1>()==1, "wrong result in layout_map at method");
    static_assert(layout1::at<2>()==2, "wrong result in layout_map at method");
    static_assert(layout1::at(0)==0, "wrong result in layout_map at method");
    static_assert(layout1::at(1)==1, "wrong result in layout_map at method");
    static_assert(layout1::at(2)==2, "wrong result in layout_map at method");

}
 
TEST(LayoutMap, ExtendedLayout) {
    typedef layout_map<3,2,1,0> layout2;

    // test length
    static_assert(layout2::masked_length==4, "layout_map length is wrong");
    static_assert(layout2::unmasked_length==4, "layout_map length is wrong");

    // test find method
    static_assert(layout2::find<0>()==3, "wrong result in layout_map find method");
    static_assert(layout2::find<1>()==2, "wrong result in layout_map find method");
    static_assert(layout2::find<2>()==1, "wrong result in layout_map find method");
    static_assert(layout2::find<3>()==0, "wrong result in layout_map find method");
    static_assert(layout2::find(0)==3, "wrong result in layout_map find method");
    static_assert(layout2::find(1)==2, "wrong result in layout_map find method");
    static_assert(layout2::find(2)==1, "wrong result in layout_map find method");
    static_assert(layout2::find(3)==0, "wrong result in layout_map find method");

    // test at method
    static_assert(layout2::at<0>()==3, "wrong result in layout_map at method");
    static_assert(layout2::at<1>()==2, "wrong result in layout_map at method");
    static_assert(layout2::at<2>()==1, "wrong result in layout_map at method");
    static_assert(layout2::at<3>()==0, "wrong result in layout_map at method");
    static_assert(layout2::at(0)==3, "wrong result in layout_map at method");
    static_assert(layout2::at(1)==2, "wrong result in layout_map at method");
    static_assert(layout2::at(2)==1, "wrong result in layout_map at method");
    static_assert(layout2::at(3)==0, "wrong result in layout_map at method");
}

TEST(LayoutMap, MaskedLayout) {
    typedef layout_map<2,-1,1,0> layout3;

    // test length
    static_assert(layout3::masked_length==4, "layout_map length is wrong");
    static_assert(layout3::unmasked_length==3, "layout_map length is wrong");;

    // test find method
    static_assert(layout3::find<0>()==3, "wrong result in layout_map find method");
    static_assert(layout3::find<1>()==2, "wrong result in layout_map find method");
    static_assert(layout3::find<2>()==0, "wrong result in layout_map find method");
    static_assert(layout3::find(0)==3, "wrong result in layout_map find method");
    static_assert(layout3::find(1)==2, "wrong result in layout_map find method");
    static_assert(layout3::find(2)==0, "wrong result in layout_map find method");

    // test at method
    static_assert(layout3::at<0>()==2, "wrong result in layout_map at method");
    static_assert(layout3::at<1>()==-1, "wrong result in layout_map at method");
    static_assert(layout3::at<2>()==1, "wrong result in layout_map at method");
    static_assert(layout3::at<3>()==0, "wrong result in layout_map at method");
    static_assert(layout3::at(0)==2, "wrong result in layout_map at method");
    static_assert(layout3::at(1)==-1, "wrong result in layout_map at method");
    static_assert(layout3::at(2)==1, "wrong result in layout_map at method");
    static_assert(layout3::at(3)==0, "wrong result in layout_map at method");
}
