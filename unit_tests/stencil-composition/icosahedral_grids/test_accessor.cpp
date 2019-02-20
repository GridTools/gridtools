/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "gtest/gtest.h"

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil-composition/accessor_metafunctions.hpp>
#include <gridtools/stencil-composition/global_accessor.hpp>
#include <gridtools/stencil-composition/icosahedral_grids/accessor.hpp>

TEST(accessor, is_accessor) {
    using namespace gridtools;
    GT_STATIC_ASSERT((is_accessor<accessor<6, intent::inout, enumtype::cells, extent<3, 4, 4, 5>>>::value), "");
    GT_STATIC_ASSERT((is_accessor<accessor<2, intent::in, enumtype::cells>>::value), "");
    GT_STATIC_ASSERT((!is_accessor<int>::value), "");
}

TEST(accessor, is_accessor_readonly) {
    using namespace gridtools;
    GT_STATIC_ASSERT((is_accessor_readonly<in_accessor<0, enumtype::cells>>::value), "");
    GT_STATIC_ASSERT((is_accessor_readonly<accessor<0, intent::in, enumtype::cells>>::value), "");
    GT_STATIC_ASSERT((is_accessor_readonly<global_accessor<0>>::value), "");
    GT_STATIC_ASSERT((!is_accessor_readonly<inout_accessor<0, enumtype::cells>>::value), "");
    GT_STATIC_ASSERT((!is_accessor_readonly<accessor<0, intent::inout, enumtype::cells>>::value), "");
}
