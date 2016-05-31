#include "gtest/gtest.h"
#include "test_offset_tuple.hpp"

using namespace gridtools;

TEST(offset_tuple, test_offset_tuple_constexpr) {

    bool result;
    test_offset_tuple(&result);
    ASSERT_TRUE(result);
}
