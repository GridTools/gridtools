#include "gtest/gtest.h"
#include <stencil-composition/stencil-composition.hpp>
#include <common/generic_metafunctions/repeat_template.hpp>

using namespace gridtools;

TEST(test_common_metafunctions, test_repeat_template) {
    typedef repeat_template_v< static_int< 5 >, static_int< 3 >, uint_t, halo >::type test1;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< test1, halo< 5, 5, 5 > >::type::value), "internal error");

    typedef typename repeat_template_v_c< 5, 3, uint_t, halo >::type test2;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< test2, halo< 5, 5, 5 > >::type::value), "internal error");
}

TEST(test_common_metafunctions, test_repeat_template_with_initial_values) {
    typedef repeat_template_v< static_int< 5 >, static_int< 3 >, uint_t, halo, 4, 7, 8 >::type test1;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< test1, halo< 4, 7, 8, 5, 5, 5 > >::type::value), "internal error");

    typedef typename repeat_template_v_c< 5, 3, uint_t, halo, 4, 7, 8 >::type test2;
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< test2, halo< 4, 7, 8, 5, 5, 5 > >::type::value), "internal error");
}
