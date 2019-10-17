/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/stencil_composition/icosahedral_grids/icosahedral_topology_metafunctions.hpp>

#include "gtest/gtest.h"

#include <gridtools/stencil_composition/location_type.hpp>

using namespace gridtools;
using namespace enumtype;

// 0(cells) + 4+8+ /*(16)*/ + 32
static_assert(impl::compute_uuid<cells::value, selector<1, 1, 0, 1>>::value ==
                  cells::value + 44 + metastorage_library_indices_limit,
    "");

// 0(cells) + 4+8+ /*(16)*/ + 32 //the rest of dimensions are ignored
static_assert(impl::compute_uuid<cells::value, selector<1, 1, 0, 1, 1, 1>>::value ==
                  cells::value + 44 + metastorage_library_indices_limit,
    "");

// 0(cells) + 4+8+ /*(16)*/ + 32 //the rest of dimensions are ignored
static_assert(impl::compute_uuid<cells::value, selector<1, 1, 1, 1, 1>>::value ==
                  cells::value + 60 + metastorage_library_indices_limit,
    "");

// 1(edges) + 4+/*8*/+ 16 + 32 //the rest of dimensions are ignored
static_assert(impl::compute_uuid<edges::value, selector<1, 0, 1, 1, 1, 1>>::value ==
                  edges::value + 52 + metastorage_library_indices_limit,
    "");

TEST(dummy, dummy) {}
