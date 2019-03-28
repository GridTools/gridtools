/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <type_traits>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/layout_map.hpp>
#include <gridtools/stencil_composition/icosahedral_grids/icosahedral_topology.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;

using top_t = icosahedral_topology<target_t>;

TEST(bakend, select_layout) {
#if defined(GT_BACKEND_X86) || defined(GT_BACKEND_NAIVE)
    GT_STATIC_ASSERT((std::is_same<top_t::layout_t<selector<1, 1, 1, 1>>, layout_map<0, 1, 2, 3>>::value), "ERROR");
    GT_STATIC_ASSERT((std::is_same<top_t::layout_t<selector<1, 0, 1, 1>>, layout_map<0, -1, 1, 2>>::value), "ERROR");
    GT_STATIC_ASSERT(
        (boost::is_same<top_t::layout_t<selector<1, 1, 0, 1, 1>>, layout_map<1, 2, -1, 3, 0>>::value), "ERROR");
#else
    GT_STATIC_ASSERT((std::is_same<top_t::layout_t<selector<1, 1, 1, 1>>, layout_map<3, 2, 1, 0>>::value), "ERROR");
    GT_STATIC_ASSERT((std::is_same<top_t::layout_t<selector<1, 0, 1, 1>>, layout_map<2, -1, 1, 0>>::value), "ERROR");
    GT_STATIC_ASSERT(
        (std::is_same<top_t::layout_t<selector<1, 1, 0, 1, 1>>, layout_map<3, 2, -1, 1, 0>>::value), "ERROR");
#endif
}
