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
#include "common/array.hpp"
#include "common/offset_tuple.hpp"
#include "common/layout_map.hpp"

GT_FUNCTION
void test_layout_accessors(bool *result) {
    using namespace gridtools;

    *result = true;
#ifdef CXX11_ENABLED
    GRIDTOOLS_STATIC_ASSERT((static_int< layout_map< 2 >::at< 0 >() >::value == 2), "Error");
    GRIDTOOLS_STATIC_ASSERT((static_int< layout_map< 1, 3 >::at< 0 >() >::value == 1), "Error");
    GRIDTOOLS_STATIC_ASSERT((static_int< layout_map< 1, 3 >::at< 1 >() >::value == 3), "Error");
    GRIDTOOLS_STATIC_ASSERT((static_int< layout_map< 1, 3, -3 >::at< 0 >() >::value == 1), "Error");
    GRIDTOOLS_STATIC_ASSERT((static_int< layout_map< 1, 3, -3 >::at< 1 >() >::value == 3), "Error");

    GRIDTOOLS_STATIC_ASSERT((static_int< layout_map< 1, 3, -3 >::at< 2 >() >::value == -3), "Error");

    GRIDTOOLS_STATIC_ASSERT((static_int< layout_map< 1, 3, -3, 5 >::at< 0 >() >::value == 1), "Error");

    GRIDTOOLS_STATIC_ASSERT((static_int< layout_map< 1, 3, -3, 5 >::at< 1 >() >::value == 3), "Error");

    GRIDTOOLS_STATIC_ASSERT((static_int< layout_map< 1, 3, -3, 5 >::at< 2 >() >::value == -3), "Error");

    GRIDTOOLS_STATIC_ASSERT((static_int< layout_map< 1, 3, -3, 5 >::at< 3 >() >::value == 5), "Error");

    ////////////////////////////////////////////////////////////////////
    {
        constexpr layout_map< 2 > lm;
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[0] >::value == 2), "Error");
    }
    {
        constexpr layout_map< 1, 3 > lm;
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[0] >::value == 1), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[1] >::value == 3), "Error");
    }
    {
        constexpr layout_map< 1, 3, -3 > lm;
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[0] >::value == 1), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[1] >::value == 3), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[2] >::value == -3), "Error");
    }
    {
        constexpr layout_map< 1, 3, -3, 5 > lm;
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[0] >::value == 1), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[1] >::value == 3), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[2] >::value == -3), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[3] >::value == 5), "Error");
    }
#endif

    *result &= ((layout_map< 2 >::at< 0 >() == 2));
    *result &= ((layout_map< 1, 3 >::at< 0 >() == 1));
    *result &= ((layout_map< 1, 3 >::at< 1 >() == 3));
    *result &= ((layout_map< 1, 3, -3 >::at< 0 >() == 1));
    *result &= ((layout_map< 1, 3, -3 >::at< 1 >() == 3));

    *result &= ((layout_map< 1, 3, -3 >::at< 2 >() == -3));

    *result &= ((layout_map< 1, 3, -3, 5 >::at< 0 >() == 1));

    *result &= ((layout_map< 1, 3, -3, 5 >::at< 1 >() == 3));

    *result &= ((layout_map< 1, 3, -3, 5 >::at< 2 >() == -3));

    *result &= ((layout_map< 1, 3, -3, 5 >::at< 3 >() == 5));

    ////////////////////////////////////////////////////////////////////

    {
        constexpr layout_map< 2 > lm;
        *result &= ((lm[0] == 2));
    }
    {
        constexpr layout_map< 1, 3 > lm;
        *result &= ((lm[0] == 1));
        *result &= ((lm[1] == 3));
    }
    {
        constexpr layout_map< 1, 3, -3 > lm;
        *result &= ((lm[0] == 1));
        *result &= ((lm[1] == 3));
        *result &= ((lm[2] == -3));
    }
    {
        constexpr layout_map< 1, 3, -3, 5 > lm;
        *result &= ((lm[0] == 1));
        *result &= ((lm[1] == 3));
        *result &= ((lm[2] == -3));
        *result &= ((lm[3] == 5));
    }

    typedef gridtools::layout_transform< gridtools::layout_map< 0, 1 >, gridtools::layout_map< 0, 1 > >::type transf0;

    *result &= ((transf0::at< 0 >() == 0));
    *result &= ((transf0::at< 1 >() == 1));
    typedef gridtools::layout_transform< gridtools::layout_map< 0, 1 >, gridtools::layout_map< 1, 0 > >::type transf01;

    *result &= ((transf01::at< 0 >() == 1));
    *result &= ((transf01::at< 1 >() == 0));
    typedef gridtools::layout_transform< gridtools::layout_map< 1, 0 >, gridtools::layout_map< 1, 0 > >::type transf02;

    *result &= ((transf02::at< 0 >() == 0));
    *result &= ((transf02::at< 1 >() == 1));
    typedef gridtools::layout_transform< gridtools::layout_map< 2, 0, 1 >, gridtools::layout_map< 2, 1, 0 > >::type
        transf;

    *result &= ((transf::at< 0 >() == 1));
    *result &= ((transf::at< 1 >() == 0));
    *result &= ((transf::at< 2 >() == 2));
    typedef gridtools::layout_transform< gridtools::layout_map< 1, 2, 0 >, gridtools::layout_map< 0, 1, 2 > >::type
        transf2;

    *result &= ((transf2::at< 0 >() == 1));
    *result &= ((transf2::at< 1 >() == 2));
    *result &= ((transf2::at< 2 >() == 0));

    int a = 10, b = 100, c = 1000;
    *result &= ((gridtools::layout_map< 2, 0, 1 >::select< 0 >(a, b, c) == c));
    *result &= ((gridtools::layout_map< 2, 0, 1 >::select< 1 >(a, b, c) == a));
    *result &= ((gridtools::layout_map< 2, 0, 1 >::select< 2 >(a, b, c) == b));
    *result &= ((gridtools::layout_map< 1, 2, 0 >::select< 0 >(a, b, c) == b));
    *result &= ((gridtools::layout_map< 1, 2, 0 >::select< 1 >(a, b, c) == c));
    *result &= ((gridtools::layout_map< 1, 2, 0 >::select< 2 >(a, b, c) == a));
    *result &= ((gridtools::layout_map< 2, 0, 1 >::find< 0 >(a, b, c) == b));
    *result &= ((gridtools::layout_map< 2, 0, 1 >::find< 1 >(a, b, c) == c));
    *result &= ((gridtools::layout_map< 2, 0, 1 >::find< 2 >(a, b, c) == a));
}

GT_FUNCTION
void test_layout_find_val(bool *result) {
    using namespace gridtools;

    *result = true;

////// TESTING FIND_VAL

#ifdef CXX11_ENABLED
    GRIDTOOLS_STATIC_ASSERT((gridtools::layout_map< 2, 0, 1 >::find_val< 0, int, 666 >(7, 9, 11) == 9), "Error");
    GRIDTOOLS_STATIC_ASSERT((gridtools::layout_map< 2, 0, 1 >::find_val< 1, int, 666 >(7, 9, 11) == 11), "Error");
    GRIDTOOLS_STATIC_ASSERT((gridtools::layout_map< 2, 0, 1 >::find_val< 2, int, 666 >(7, 9, 11) == 7), "Error");
    GRIDTOOLS_STATIC_ASSERT((gridtools::layout_map< 2, 0, 1 >::find_val< 3, int, 666 >(7, 9, 11) == 666), "Error");

    GRIDTOOLS_STATIC_ASSERT((static_int< gridtools::layout_map< 2, 0, 1 >::find_val< 0, int, 666 >(
                                    offset_tuple< 3, 3 >(7, 9, 11)) >::value == 9),
        "Error");

    GRIDTOOLS_STATIC_ASSERT((static_int< gridtools::layout_map< 2, 0, 1 >::find_val< 1, int, 666 >(
                                    offset_tuple< 3, 3 >(7, 9, 11)) >::value == 11),
        "Error");

    GRIDTOOLS_STATIC_ASSERT((static_int< gridtools::layout_map< 2, 0, 1 >::find_val< 2, int, 666 >(
                                    offset_tuple< 3, 3 >(7, 9, 11)) >::value == 7),
        "Error");

    GRIDTOOLS_STATIC_ASSERT((static_int< gridtools::layout_map< 2, 0, 1 >::find_val< 3, int, 666 >(
                                    offset_tuple< 3, 3 >(7, 9, 11)) >::value == 666),
        "Error");

#ifndef __CUDACC__
    GRIDTOOLS_STATIC_ASSERT(
        (static_int< gridtools::layout_map< 2, 0, 1 >::find_val< 0, int, 666 >(array< uint_t, 3 >{7, 9, 11}) >::value ==
            9),
        "Error");
    GRIDTOOLS_STATIC_ASSERT(
        (static_int< gridtools::layout_map< 2, 0, 1 >::find_val< 1, int, 666 >(array< uint_t, 3 >{7, 9, 11}) >::value ==
            11),
        "Error");
    GRIDTOOLS_STATIC_ASSERT(
        (static_int< gridtools::layout_map< 2, 0, 1 >::find_val< 2, int, 666 >(array< uint_t, 3 >{7, 9, 11}) >::value ==
            7),
        "Error");
    GRIDTOOLS_STATIC_ASSERT(
        (static_int< gridtools::layout_map< 2, 0, 1 >::find_val< 3, int, 666 >(array< uint_t, 3 >{7, 9, 11}) >::value ==
            666),
        "Error");
#endif

#endif

    *result &= ((layout_map< 2, 0, 1 >::find_val< 0, int, 666 >(7, 9, 11) == 9));
    *result &= ((layout_map< 2, 0, 1 >::find_val< 1, int, 666 >(7, 9, 11) == 11));
    *result &= ((layout_map< 2, 0, 1 >::find_val< 2, int, 666 >(7, 9, 11) == 7));
    *result &= ((layout_map< 2, 0, 1 >::find_val< 3, int, 666 >(7, 9, 11) == 666));
    *result &= ((layout_map< 2, 0, 1 >::find_val< 0, int, 666 >(offset_tuple< 3, 3 >(7, 9, 11)) == 9));
    *result &= ((layout_map< 2, 0, 1 >::find_val< 1, int, 666 >(offset_tuple< 3, 3 >(7, 9, 11)) == 11));
    *result &= ((layout_map< 2, 0, 1 >::find_val< 2, int, 666 >(offset_tuple< 3, 3 >(7, 9, 11)) == 7));

#ifdef CXX11_ENABLED
    // TODO: the find_val method does not return default value with cxx03
    *result &= ((layout_map< 2, 0, 1 >::find_val< 3, int, 666 >(offset_tuple< 3, 3 >(7, 9, 11)) == 666));
    *result &= ((layout_map< 2, 0, 1 >::find_val< 0, int, 666 >(array< uint_t, 3 >{7, 9, 11}) == 9));
    *result &= ((layout_map< 2, 0, 1 >::find_val< 1, int, 666 >(array< uint_t, 3 >{7, 9, 11}) == 11));
    *result &= ((layout_map< 2, 0, 1 >::find_val< 2, int, 666 >(array< uint_t, 3 >{7, 9, 11}) == 7));
    *result &= ((layout_map< 2, 0, 1 >::find_val< 3, int, 666 >(array< uint_t, 3 >{7, 9, 11}) == 666));
#endif
}
