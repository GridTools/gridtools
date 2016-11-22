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
#include "common/generic_metafunctions/accumulate.hpp"
#include "common/generic_metafunctions/logical_ops.hpp"
#include "common/array.hpp"

using namespace gridtools;

template < typename... Args >
GT_FUNCTION static constexpr bool check_or(Args... args) {
    return accumulate(logical_or(), is_array< Args >::type::value...);
}

template < typename... Args >
GT_FUNCTION static constexpr bool check_and(Args... args) {
    return accumulate(logical_and(), is_array< Args >::type::value...);
}

GT_FUNCTION
static bool test_accumulate_and() {
    GRIDTOOLS_STATIC_ASSERT((check_and(array< uint_t, 4 >{3, 4, 5, 6}, array< int_t, 2 >{-2, 3})), "Error");
    GRIDTOOLS_STATIC_ASSERT((!check_and(array< uint_t, 4 >{3, 4, 5, 6}, array< int_t, 2 >{-2, 3}, 7)), "Error");

    return true;
}

GT_FUNCTION
static bool test_accumulate_or() {

    GRIDTOOLS_STATIC_ASSERT((check_or(array< uint_t, 4 >{3, 4, 5, 6}, array< int_t, 2 >{-2, 3})), "Error");
    GRIDTOOLS_STATIC_ASSERT((check_or(array< uint_t, 4 >{3, 4, 5, 6}, array< int_t, 2 >{-2, 3})), "Error");
    GRIDTOOLS_STATIC_ASSERT((check_or(array< uint_t, 4 >{3, 4, 5, 6}, array< int_t, 2 >{-2, 3}, 7)), "Error");
    GRIDTOOLS_STATIC_ASSERT((!check_or(-2, 3, 7)), "Error");

    return true;
}
