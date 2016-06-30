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
#include "gtest/gtest.h"
#include "defs.hpp"
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
