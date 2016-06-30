#include "gtest/gtest.h"
#include "test_layout_map.hpp"

using namespace gridtools;

TEST(layout_map, accessors) {
    bool result = true;
    test_layout_accessors(&result);
    ASSERT_TRUE(result);
}

TEST(layout_map, find_val) {
    bool result = true;
    test_layout_find_val(&result);

    ASSERT_TRUE(&result);
}
