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
#include <stencil-composition/stencil-composition.hpp>
#include <common/generic_metafunctions/repeat_template.hpp>

using namespace gridtools;

TEST(test_common_metafunctions, test_repeat_template) {
    typedef repeat_template_v< static_int< 5 >, static_int< 3 >, uint_t, halo >::type test1;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< test1, halo< 5, 5, 5 > >::type::value), "internal error");

    typedef typename repeat_template_v_c< 5, 3, uint_t, halo >::type test2;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< test2, halo< 5, 5, 5 > >::type::value), "internal error");
}

TEST(test_common_metafunctions, test_repeat_template_with_initial_values) {
    typedef repeat_template_v< static_int< 5 >, static_int< 3 >, uint_t, halo, 4, 7, 8 >::type test1;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< test1, halo< 4, 7, 8, 5, 5, 5 > >::type::value), "internal error");

    typedef typename repeat_template_v_c< 5, 3, uint_t, halo, 4, 7, 8 >::type test2;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< test2, halo< 4, 7, 8, 5, 5, 5 > >::type::value), "internal error");
}
