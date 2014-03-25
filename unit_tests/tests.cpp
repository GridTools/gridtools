#include "gtest/gtest.h"

#include "test_domain_indices.h"

TEST(testdomain, testindices) {
    EXPECT_EQ(test_domain_indices(), true);
}



int main(int argc, char** argv) {

    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
