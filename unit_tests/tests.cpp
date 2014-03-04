#include "gtest/gtest.h"

#include "test_domain.h"

TEST(testdomain, testallocationongpu) {
    EXPECT_EQ(test_domain(), false);
}

int main(int argc, char** argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
