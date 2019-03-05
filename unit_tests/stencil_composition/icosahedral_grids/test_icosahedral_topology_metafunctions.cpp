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
#include <gridtools/common/gt_assert.hpp>
#include <gridtools/stencil_composition/icosahedral_grids/icosahedral_topology_metafunctions.hpp>
#include <gridtools/stencil_composition/location_type.hpp>

using namespace gridtools;

TEST(icosahedral_topology_metafunctions, selector_uuid) {
    // 0(cells) + 4+8+ /*(16)*/ + 32
    GT_STATIC_ASSERT((impl::compute_uuid<enumtype::cells::value, selector<1, 1, 0, 1>>::value ==
                         enumtype::cells::value + 44 + metastorage_library_indices_limit),
        "ERROR");

    // 0(cells) + 4+8+ /*(16)*/ + 32 //the rest of dimensions are ignored
    GT_STATIC_ASSERT((impl::compute_uuid<enumtype::cells::value, selector<1, 1, 0, 1, 1, 1>>::value ==
                         enumtype::cells::value + 44 + metastorage_library_indices_limit),
        "ERROR");

    // 0(cells) + 4+8+ /*(16)*/ + 32 //the rest of dimensions are ignored
    GT_STATIC_ASSERT((impl::compute_uuid<enumtype::cells::value, selector<1, 1, 1, 1, 1>>::value ==
                         enumtype::cells::value + 60 + metastorage_library_indices_limit),
        "ERROR");

    // 1(edges) + 4+/*8*/+ 16 + 32 //the rest of dimensions are ignored
    GT_STATIC_ASSERT((impl::compute_uuid<enumtype::edges::value, selector<1, 0, 1, 1, 1, 1>>::value ==
                         enumtype::edges::value + 52 + metastorage_library_indices_limit),
        "ERROR");
}
