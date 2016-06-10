#include "gtest/gtest.h"
#include <stencil_composition/stencil_composition.hpp>
#include <common/generic_metafunctions/repeat_template.hpp>

using namespace gridtools;
TEST(test_common_metafunctions, test_repeat_template){
    typedef typename repeat_template<static_int<5>, static_int<3>, halo>::type test1;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same<test1, halo<5,5,5> >::type::value), "internal error");

    typedef typename repeat_template_c<5, 3, halo>::type test2;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same<test2, halo<5,5,5> >::type::value), "internal error");
}
