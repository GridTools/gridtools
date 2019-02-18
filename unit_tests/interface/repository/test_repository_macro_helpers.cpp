/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/interface/repository/repository_macro_helpers.hpp>
#include <gtest/gtest.h>

TEST(repository_macros, max_in_tuple) {
#define my_tuple (0, 1, 4)
    ASSERT_EQ(4, GT_REPO_max_in_tuple(my_tuple));
#undef my_tuple
}

TEST(repository_macros, max_dim) {
#define my_field_types (IJKDataStore, (0, 1, 5))(IJDataStore, (0, 1))(AnotherDataStore, (8, 1))
    int result = GT_REPO_max_dim(BOOST_PP_VARIADIC_SEQ_TO_SEQ(my_field_types));
    ASSERT_EQ(8, result);
#undef my_field_types
}

TEST(repository_macros, has_dim) {
#define my_field_types (IJKDataStore, (0, 1, 5))(IJDataStore, (0, 1))(AnotherDataStore, (8, 1))
    ASSERT_GT(GT_REPO_has_dim(BOOST_PP_VARIADIC_SEQ_TO_SEQ(my_field_types)), 0);
#undef my_field_types
#define my_field_types (IJKDataStore)(IJDataStore)(AnotherDataStore)
    ASSERT_EQ(0, GT_REPO_has_dim(BOOST_PP_VARIADIC_SEQ_TO_SEQ(my_field_types)));
#undef my_field_types
}
