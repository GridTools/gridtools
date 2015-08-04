
#include "gtest/gtest.h"
#include "defs.hpp"
#include "common/generic_metafunctions/is_not_same.hpp"

using namespace gridtools;

TEST(is_not_same, test)
{
    GRIDTOOLS_STATIC_ASSERT((is_not_same<int, float>::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((! is_not_same<int, int>::value),"ERROR");
    GRIDTOOLS_STATIC_ASSERT((is_not_same<double, float>::value),"ERROR");

    ASSERT_TRUE(true);
}


