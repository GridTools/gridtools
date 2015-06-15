#define PEDANTIC_DISABLED // to stringent for this test
#include "gtest/gtest.h"
#include "test_iterate_domain.h"

TEST(testdomain, iterate_domain) {
    EXPECT_EQ(test_iterate_domain::test(), true);
}
