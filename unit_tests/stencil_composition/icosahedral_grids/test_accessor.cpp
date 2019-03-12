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
#include <gridtools/stencil_composition/global_accessor.hpp>
#include <gridtools/stencil_composition/icosahedral_grids/accessor.hpp>

TEST(accessor, is_accessor) {
    using namespace gridtools;
    GT_STATIC_ASSERT((is_accessor<accessor<6, intent::inout, enumtype::cells, extent<3, 4, 4, 5>>>::value), "");
    GT_STATIC_ASSERT((is_accessor<accessor<2, intent::in, enumtype::cells>>::value), "");
    GT_STATIC_ASSERT((!is_accessor<int>::value), "");
}
