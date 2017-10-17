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

#include <common/layout_map_metafunctions.hpp>
#include <common/gt_assert.hpp>

using namespace gridtools;

TEST(layout_map_metafunctions, filter_layout) {

    {
        using layout_map_t = layout_map< 0, 1, 2, 3 >;
        using filtered_layout_map_t = filter_layout< layout_map_t, selector< 1, 1, 0, 1 > >::type;
        GRIDTOOLS_STATIC_ASSERT((boost::is_same< filtered_layout_map_t, layout_map< 0, 1, -1, 2 > >::value), "Error");
    }
    {
        using layout_map_t = layout_map< 3, 1, 2, 0 >;
        using filtered_layout_map_t = filter_layout< layout_map_t, selector< 1, 0, 0, 1 > >::type;
        GRIDTOOLS_STATIC_ASSERT((boost::is_same< filtered_layout_map_t, layout_map< 1, -1, -1, 0 > >::value), "Error");
    }
}
TEST(layout_map_metafunctions, extend_layout_map) {
    {
        using layout_map_t = layout_map< 0, 1, 2, 3 >;
        using extended_layout_map_t = extend_layout_map< layout_map_t, 3 >::type;
        GRIDTOOLS_STATIC_ASSERT(
            (boost::is_same< extended_layout_map_t, layout_map< 3, 4, 5, 6, 0, 1, 2 > >::value), "Error");
    }
    {
        using layout_map_t = layout_map< 3, 2, 1, 0 >;
        using extended_layout_map_t = extend_layout_map< layout_map_t, 3 >::type;
        GRIDTOOLS_STATIC_ASSERT(
            (boost::is_same< extended_layout_map_t, layout_map< 6, 5, 4, 3, 0, 1, 2 > >::value), "Error");
    }
    {
        using layout_map_t = layout_map< 3, 1, 0, 2 >;
        using extended_layout_map_t = extend_layout_map< layout_map_t, 3 >::type;
        GRIDTOOLS_STATIC_ASSERT(
            (boost::is_same< extended_layout_map_t, layout_map< 6, 4, 3, 5, 0, 1, 2 > >::value), "Error");
    }
}
