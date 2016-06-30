/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
