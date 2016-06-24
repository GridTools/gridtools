#include "gtest/gtest.h"
#include "test_accumulate.hpp"

TEST(accumulate, test_and) {
    ASSERT_TRUE(test_accumulate_and());
}

TEST(accumulate, test_or) {
    ASSERT_TRUE(test_accumulate_or());
}
