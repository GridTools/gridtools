/*
 * test_smallstorage_indices.cpp
 *
 *  Created on: Jul 21, 2015
 *      Author: cosuna
 */

#include "gtest/gtest.h"
#include "common/layout_map.hpp"
#include "storage/small_storage.hpp"

TEST(smallstorage, indices) {

    typedef gridtools::layout_map<1,0,2> layout;

    gridtools::small_storage<int, layout, 15, 10, 5> x;

    bool result = x._index(1,0,0) == 5;
    result = result && (x._index(0,1,0) == 75);
    result = result && (x._index(0,0,1) == 1);

    EXPECT_EQ(result, true);
}
