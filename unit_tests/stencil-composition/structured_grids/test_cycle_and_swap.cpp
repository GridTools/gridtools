#include "test_cycle_and_swap.hpp"

#include "gtest/gtest.h"

TEST(cycle_and_swap, 2D){
    EXPECT_TRUE(test_cycle_and_swap::test_2D());
}

TEST(cycle_and_swap, 3D){
    EXPECT_TRUE(test_cycle_and_swap::test_3D());
}
