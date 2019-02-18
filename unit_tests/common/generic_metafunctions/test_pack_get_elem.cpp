/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "gtest/gtest.h"

#include <gridtools/common/generic_metafunctions/pack_get_elem.hpp>
#include <gridtools/common/gt_assert.hpp>

using namespace gridtools;

TEST(pack_get_elem, test) {
    GT_STATIC_ASSERT((pack_get_elem<2>::apply(3, 6, 7) == 7), "ERROR");
    GT_STATIC_ASSERT((pack_get_elem<1>::apply(-3, -6, 7) == -6), "ERROR");
}
