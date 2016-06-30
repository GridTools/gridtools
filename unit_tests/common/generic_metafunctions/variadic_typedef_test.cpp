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
#define BOOST_NO_CXX11_RVALUE_REFERENCES

#include "gtest/gtest.h"

#include "common/generic_metafunctions/variadic_typedef.hpp"

using namespace gridtools;

#ifdef NDEBUG
TEST(variadic_typedef, test) {

    typedef variadic_typedef< int, double, unsigned int > tt;

    GRIDTOOLS_STATIC_ASSERT((boost::is_same< tt::template get_elem< 0 >::type, int >::value), "Error");

    GRIDTOOLS_STATIC_ASSERT((boost::is_same< tt::template get_elem< 1 >::type, double >::value), "Error");

    GRIDTOOLS_STATIC_ASSERT((boost::is_same< tt::template get_elem< 2 >::type, unsigned int >::value), "Error");

    ASSERT_TRUE(true);
}

TEST(variadic_typedef, get_from_variadic_pack) {

    GRIDTOOLS_STATIC_ASSERT((static_int< get_from_variadic_pack< 3 >::apply(2, 6, 8, 3, 5) >::value == 3), "Error");

    GRIDTOOLS_STATIC_ASSERT(
        (static_int< get_from_variadic_pack< 7 >::apply(2, 6, 8, 3, 5, 4, 6, -8, 4, 3, 1, 54, 67) >::value == -8),
        "Error");

    ASSERT_TRUE(true);
}
#endif
