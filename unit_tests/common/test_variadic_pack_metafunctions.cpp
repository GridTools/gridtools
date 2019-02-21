/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "gtest/gtest.h"
#include <gridtools/common/defs.hpp>
#include <gridtools/common/variadic_pack_metafunctions.hpp>

TEST(VariadicPackMetafunctions, GetIndexOfElementInVariadicPack) {
    GT_STATIC_ASSERT((gridtools::get_index_of_element_in_pack(0, 1, 1, 2, 3, 4) == 0),
        "Failed to retrieve correct index from varidadic pack.");
    GT_STATIC_ASSERT((gridtools::get_index_of_element_in_pack(0, 2, 1, 2, 3, 4) == 1),
        "Failed to retrieve correct index from varidadic pack.");
    GT_STATIC_ASSERT((gridtools::get_index_of_element_in_pack(0, 3, 1, 2, 3, 4) == 2),
        "Failed to retrieve correct index from varidadic pack.");
    GT_STATIC_ASSERT((gridtools::get_index_of_element_in_pack(0, 4, 1, 2, 3, 4) == 3),
        "Failed to retrieve correct index from varidadic pack.");
}

TEST(VariadicPackMetafunctions, GetElementFromVariadicPack) {
    GT_STATIC_ASSERT(
        (gridtools::get_value_from_pack(0, 1, 2, 3, 4) == 1), "Failed to retrieve correct value from varidadic pack.");
    GT_STATIC_ASSERT(
        (gridtools::get_value_from_pack(1, 1, 2, 3, 4) == 2), "Failed to retrieve correct value from varidadic pack.");
    GT_STATIC_ASSERT(
        (gridtools::get_value_from_pack(2, 1, 2, 3, 4) == 3), "Failed to retrieve correct value from varidadic pack.");
    GT_STATIC_ASSERT(
        (gridtools::get_value_from_pack(3, 1, 2, 3, 4) == 4), "Failed to retrieve correct value from varidadic pack.");
}
