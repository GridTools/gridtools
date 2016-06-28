#include "gtest/gtest.h"
#include "common/defs.hpp"
#include "common/array.hpp"
#include "common/array_addons.hpp"

using namespace gridtools;

TEST(array, test_append) {
    array<uint_t, 4> a{1,2,3,4};
    auto mod_a = a.append_dim(5);
    ASSERT_TRUE((mod_a == array<uint_t, 5>{1,2,3,4,5}));
    ASSERT_TRUE((mod_a[4] == 5));
}

TEST(array, test_prepend) {
    constexpr array<uint_t, 4> a{1,2,3,4};
    auto mod_a = a.prepend_dim(5);
    ASSERT_TRUE((mod_a == array<uint_t, 5>{5,1,2,3,4}));
    ASSERT_TRUE((mod_a[0] == 5));
}

TEST(array, test_copyctr) {
    constexpr array<uint_t, 4> a{4,2,3,1};
    constexpr auto mod_a(a);
    ASSERT_TRUE((mod_a == array<uint_t, 4>{4,2,3,1}));
    ASSERT_TRUE((mod_a[0] == 4));
}
