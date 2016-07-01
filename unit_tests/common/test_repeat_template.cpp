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
