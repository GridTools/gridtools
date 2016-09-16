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
#pragma once
#include "common/defs.hpp"
#include "common/tuple.hpp"

#ifdef CXX11_ENABLED
GT_FUNCTION
void test_tuple_elements(bool *result) {
    using namespace gridtools;

    *result = true;
    constexpr tuple< int_t, short_t, uint_t > tup(-3, 4, 10);

    GRIDTOOLS_STATIC_ASSERT((static_int< tup.get< 0 >() >::value == -3), "ERROR");

#if defined(CXX11_ENABLED) && !defined(__CUDACC__)

    // CUDA does not think the following are constexprable :(
    GRIDTOOLS_STATIC_ASSERT((static_int< tup.n_dimensions >::value == 3), "ERROR");
    GRIDTOOLS_STATIC_ASSERT((static_int< tup.get< 1 >() >::value == 4), "ERROR");
    GRIDTOOLS_STATIC_ASSERT((static_int< tup.get< 2 >() >::value == 10), "ERROR");
#endif

    *result &= ((tup.get< 0 >() == -3));
    *result &= ((tup.get< 1 >() == 4));
    *result &= ((tup.get< 2 >() == 10));
}

#endif
