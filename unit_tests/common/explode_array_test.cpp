#include "gtest/gtest.h"

#include "explode_array_test.hpp"

TEST(explode_array, test_explode_static) {
    ASSERT_TRUE(test_explode_static());
}

TEST(explode_array, test_explode_with_object) {
    ASSERT_TRUE(test_explode_with_object());
}

TEST(explode_array, tuple) {
    ASSERT_TRUE((test_explode_with_tuple()));
}

TEST(explode_array, tuple_with_object) {
    ASSERT_TRUE((test_explode_with_tuple_with_object()));
}
