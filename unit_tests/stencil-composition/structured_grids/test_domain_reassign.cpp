#include "gtest/gtest.h"
#include "test_domain_reassign.hpp"
#include "../../examples/Options.hpp"

int main(int argc, char **argv) {

    // Pass command line arguments to googltest
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

TEST(ReassignDomain, Test) { ASSERT_TRUE(domain_reassign::test()); }
