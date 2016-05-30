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

    typedef meta_array<boost::mpl::vector4<int, int, int, int> , boost::mpl::quote1<boost::is_integral> > meta_array_t;

    ASSERT_TRUE(( boost::mpl::equal<meta_array_t::elements, boost::mpl::vector4<int, int, int, int> >::value ));

    GRIDTOOLS_STATIC_ASSERT(( is_meta_array_of< meta_array_t, boost::is_integral>::value ), "Error");
}
