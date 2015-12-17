#include "test_cycle_and_swap.hpp"

#include "gtest/gtest.h"


#ifdef CUDA_EXAMPLE
#define BACKEND backend<Cuda, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend<Host, Block >
#else
#define BACKEND backend<Host, Naive >
#endif
#endif

TEST(cycle_and_swap, 2D){
    EXPECT_TRUE(test_cycle_and_swap::test_2D());
}

TEST(cycle_and_swap, 3D){
    EXPECT_TRUE(test_cycle_and_swap::test_3D());
}
