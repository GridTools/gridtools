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
#include "boost/mpl/quote.hpp"
#include "boost/mpl/vector.hpp"
#include "boost/type_traits/is_integral.hpp"
#include "common/defs.hpp"
#include <stencil-composition/stencil-composition.hpp>
#include "common/meta_array.hpp"

using namespace gridtools;
template<typename T> struct is_integer : boost::mpl::false_{};
template<> struct is_integer<int> : boost::mpl::true_{};

TEST(meta_array, test_meta_array_element) {

    typedef meta_array< boost::mpl::vector4< int, int, int, int >, boost::mpl::quote1< boost::is_integral > >
        meta_array_t;

    ASSERT_TRUE(( boost::mpl::equal<meta_array_t::elements, boost::mpl::vector4<int, int, int, int> >::value ));

    GRIDTOOLS_STATIC_ASSERT((is_meta_array_of< meta_array_t, boost::is_integral >::value), "Error");
}
