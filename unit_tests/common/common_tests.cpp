#include "gtest/gtest.h"

#include "meta_array_test.hpp"

TEST(meta_array, test_meta_array_elements) {
    EXPECT_EQ(test_meta_array_elements(), true);
}

TEST(meta_array, is_meta_array_of)
{
    EXPECT_EQ(test_is_meta_array_of(), true);
}

int main(int argc, char** argv)
{

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
