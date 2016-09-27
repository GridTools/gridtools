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

// #include <boost/mpl/map/map10.hpp>
#include <stencil-composition/structured_grids/accessor.hpp>
#include <stencil-composition/structured_grids/accessor_metafunctions.hpp>
// #include "stencil-composition/iterate_domain_remapper.hpp"

TEST(accessor, is_accessor) {
    using namespace gridtools;
    GRIDTOOLS_STATIC_ASSERT((is_accessor< accessor< 6, enumtype::inout, extent< 3, 4, 4, 5 > > >::value) == true, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< accessor< 2, enumtype::in > >::value) == true, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< int >::value) == false, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< double & >::value) == false, "");
    GRIDTOOLS_STATIC_ASSERT((is_accessor< double const & >::value) == false, "");
}

TEST(accessor, copy_const) {

    // using namespace gridtools;
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
