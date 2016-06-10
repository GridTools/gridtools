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

#include <boost/type_traits/is_integral.hpp>
#include <boost/mpl/vector/vector10.hpp>
#include <boost/mpl/equal.hpp>
#include <stencil_composition/stencil_composition.hpp>
#include "common/meta_array.hpp"

template<typename T> struct is_integer : boost::mpl::false_{};
template<> struct is_integer<int> : boost::mpl::true_{};

TEST(meta_array, test_meta_array_element) {

    typedef gridtools::meta_array<boost::mpl::vector4<int, int, int, int> , boost::mpl::quote1<is_integer> > meta_array_t;

    ASSERT_TRUE(( boost::mpl::equal<meta_array_t::elements, boost::mpl::vector4<int, int, int, int> >::value ));

    ASSERT_TRUE(( gridtools::is_meta_array_of< meta_array_t, is_integer>::value ));
}
