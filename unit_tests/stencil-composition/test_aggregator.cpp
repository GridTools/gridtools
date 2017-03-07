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
#include "common/defs.hpp"
#include "stencil-composition/arg.hpp"
#include "stencil-composition/aggregator_type.hpp"

using namespace gridtools;
using namespace enumtype;

TEST(AggregatorType, ContinousIndicesTest) {
    typedef arg< 0, int_t > arg0_t;
    typedef arg< 1, int_t > arg1_t;
    typedef arg< 2, int_t > arg2_t;
    typedef arg< 3, int_t > arg3_t;

    ASSERT_TRUE((_impl::continious_indices_check< boost::mpl::vector< arg0_t, arg1_t, arg2_t, arg3_t > >::type::value));
    ASSERT_TRUE((_impl::continious_indices_check< boost::mpl::vector< arg0_t, arg1_t, arg2_t > >::type::value));
    ASSERT_TRUE((_impl::continious_indices_check< boost::mpl::vector< arg0_t, arg1_t > >::type::value));
    ASSERT_TRUE((_impl::continious_indices_check< boost::mpl::vector< arg0_t > >::type::value));

    typedef typename boost::mpl::sort< boost::mpl::vector< arg3_t, arg2_t, arg0_t, arg1_t >, arg_comparator >::type
        placeholders_t;
    ASSERT_TRUE((_impl::continious_indices_check< placeholders_t >::type::value));
    ASSERT_FALSE((_impl::continious_indices_check< boost::mpl::vector< arg1_t, arg3_t, arg2_t > >::type::value));
    ASSERT_FALSE((_impl::continious_indices_check< boost::mpl::vector< arg0_t, arg2_t > >::type::value));
    ASSERT_FALSE((_impl::continious_indices_check< boost::mpl::vector< arg1_t > >::type::value));
}
