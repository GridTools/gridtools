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
