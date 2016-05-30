#include "gtest/gtest.h"
#include "common/defs.hpp"
#include "common/array.hpp"
#include "common/offset_tuple.hpp"
#include "common/layout_map.hpp"

using namespace gridtools;

TEST(layout_map, accessors) {

    ASSERT_TRUE((gridtools::layout_map< 2 >::at< 0 >() == 2));
    ASSERT_TRUE((gridtools::layout_map< 1, 3 >::at< 0 >() == 1));
    ASSERT_TRUE((gridtools::layout_map< 1, 3 >::at< 1 >() == 3));
    ASSERT_TRUE((gridtools::layout_map< 1, 3, -3 >::at< 0 >() == 1));
    ASSERT_TRUE((gridtools::layout_map< 1, 3, -3 >::at< 1 >() == 3));

    ASSERT_TRUE((gridtools::layout_map< 1, 3, -3 >::at< 2 >() == -3));

    ASSERT_TRUE((gridtools::layout_map< 1, 3, -3, 5 >::at< 0 >() == 1));

    ASSERT_TRUE((gridtools::layout_map< 1, 3, -3, 5 >::at< 1 >() == 3));

    ASSERT_TRUE((gridtools::layout_map< 1, 3, -3, 5 >::at< 2 >() == -3));

    ASSERT_TRUE((gridtools::layout_map< 1, 3, -3, 5 >::at< 3 >() == 5));

    ////////////////////////////////////////////////////////////////////
    /// \brief ASSERT_TRUE
    {
        constexpr layout_map< 2 > lm;
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[0] >::value == 2), "Error");
    }
    {
        constexpr layout_map<1,3> lm;
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[0] >::value == 1), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[1] >::value == 3), "Error");
    }
    {
        constexpr layout_map<1,3,-3> lm;
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[0] >::value == 1), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[1] >::value == 3), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[2] >::value == -3), "Error");
    }
    {
        constexpr layout_map<1,3,-3,5> lm;
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[0] >::value == 1), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[1] >::value == 3), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[2] >::value == -3), "Error");
        GRIDTOOLS_STATIC_ASSERT((static_short< lm[3] >::value == 5), "Error");
    }

    typedef gridtools::layout_transform< gridtools::layout_map< 0, 1 >, gridtools::layout_map< 0, 1 > >::type transf0;

    ASSERT_TRUE((transf0::at< 0 >() == 0));
    ASSERT_TRUE((transf0::at< 1 >() == 1));
    typedef gridtools::layout_transform< gridtools::layout_map< 0, 1 >, gridtools::layout_map< 1, 0 > >::type transf01;

    ASSERT_TRUE((transf01::at< 0 >() == 1));
    ASSERT_TRUE((transf01::at< 1 >() == 0));
    typedef gridtools::layout_transform< gridtools::layout_map< 1, 0 >, gridtools::layout_map< 1, 0 > >::type transf02;

    ASSERT_TRUE((transf02::at< 0 >() == 0));
    ASSERT_TRUE((transf02::at< 1 >() == 1));
    typedef gridtools::layout_transform< gridtools::layout_map< 2, 0, 1 >, gridtools::layout_map< 2, 1, 0 > >::type
        transf;

    ASSERT_TRUE((transf::at< 0 >() == 1));
    ASSERT_TRUE((transf::at< 1 >() == 0));
    ASSERT_TRUE((transf::at< 2 >() == 2));
    typedef gridtools::layout_transform< gridtools::layout_map< 1, 2, 0 >, gridtools::layout_map< 0, 1, 2 > >::type
        transf2;

    ASSERT_TRUE((transf2::at< 0 >() == 1));
    ASSERT_TRUE((transf2::at< 1 >() == 2));
    ASSERT_TRUE((transf2::at< 2 >() == 0));

    int a = 10, b = 100, c = 1000;
    ASSERT_TRUE((gridtools::layout_map< 2, 0, 1 >::select< 0 >(a, b, c) == c));
    ASSERT_TRUE((gridtools::layout_map< 2, 0, 1 >::select< 1 >(a, b, c) == a));
    ASSERT_TRUE((gridtools::layout_map< 2, 0, 1 >::select< 2 >(a, b, c) == b));
    ASSERT_TRUE((gridtools::layout_map< 1, 2, 0 >::select< 0 >(a, b, c) == b));
    ASSERT_TRUE((gridtools::layout_map< 1, 2, 0 >::select< 1 >(a, b, c) == c));
    ASSERT_TRUE((gridtools::layout_map< 1, 2, 0 >::select< 2 >(a, b, c) == a));
    ASSERT_TRUE((gridtools::layout_map< 2, 0, 1 >::find< 0 >(a, b, c) == b));
    ASSERT_TRUE((gridtools::layout_map< 2, 0, 1 >::find< 1 >(a, b, c) == c));
    ASSERT_TRUE((gridtools::layout_map< 2, 0, 1 >::find< 2 >(a, b, c) == a));
}
TEST(layout_map, find_val) {
    int a = 10, b = 100, c = 1000;
    ////// TESTING FIND_VAL
    ASSERT_TRUE((gridtools::layout_map< 2, 0, 1 >::find_val< 0, int, 666 >(a, b, c) == b));
    ASSERT_TRUE((gridtools::layout_map< 2, 0, 1 >::find_val< 1, int, 666 >(a, b, c) == c));
    ASSERT_TRUE((gridtools::layout_map< 2, 0, 1 >::find_val< 2, int, 666 >(a, b, c) == a));
    ASSERT_TRUE((gridtools::layout_map< 2, 0, 1 >::find_val< 3, int, 666 >(a, b, c) == 666));
}
