#include "gtest/gtest.h"
#include "test_tuple.hpp"

TEST(tuple, test_tuple) {
    bool result = true;
    test_tuple_elements(&result);

    ASSERT_TRUE(result);
}

