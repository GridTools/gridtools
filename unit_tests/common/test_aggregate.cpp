#include "gtest/gtest.h"
#include "common/is_aggregate.hpp"
#include "common/array.hpp"
#include "storage/wrap_pointer.hpp"

using namespace gridtools;


TEST(array, test_is_aggregate) {
    GRIDTOOLS_STATIC_ASSERT((is_aggregate<int>::value), "Error");

    typedef array<uint_t, 4> array_t;
    GRIDTOOLS_STATIC_ASSERT((is_aggregate<array_t>::value), "Error");

    typedef wrap_pointer<double> ptr_t;
    GRIDTOOLS_STATIC_ASSERT((! is_aggregate<ptr_t>::value), "Error");

    ASSERT_TRUE(true);
}

