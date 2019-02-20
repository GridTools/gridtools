/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/stencil-composition/icosahedral_grids/icosahedral_topology.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;

using icosahedral_topology_t = icosahedral_topology<backend_t>;
TEST(icosahedral_topology, layout) {
    using alayout_t = icosahedral_topology_t::layout_t<selector<1, 1, 1, 1>>;
#ifdef __CUDACC__
    GT_STATIC_ASSERT((boost::is_same<alayout_t, layout_map<3, 2, 1, 0>>::value), "ERROR");
#else
    GT_STATIC_ASSERT((boost::is_same<alayout_t, layout_map<0, 1, 2, 3>>::value), "ERROR");
#endif

    using alayout_2d_t = icosahedral_topology_t::layout_t<selector<1, 1, 1, 0>>;
#ifdef __CUDACC__
    GT_STATIC_ASSERT((boost::is_same<alayout_2d_t, layout_map<2, 1, 0, -1>>::value), "ERROR");
#else
    GT_STATIC_ASSERT((boost::is_same<alayout_2d_t, layout_map<0, 1, 2, -1>>::value), "ERROR");
#endif

    using alayout_6d_t = icosahedral_topology_t::layout_t<selector<1, 1, 1, 1, 1, 1>>;
#ifdef __CUDACC__
    GT_STATIC_ASSERT((boost::is_same<alayout_6d_t, layout_map<5, 4, 3, 2, 1, 0>>::value), "ERROR");
#else
    GT_STATIC_ASSERT((boost::is_same<alayout_6d_t, layout_map<2, 3, 4, 5, 0, 1>>::value), "ERROR");
#endif
}

TEST(icosahedral_topology, make_storage) {

    icosahedral_topology_t grid(4, 6, 7);
    icosahedral_topology_t::meta_storage_t<icosahedral_topology_t::edges, halo<0, 0, 0, 0>, selector<1, 1, 1, 1>> x(
        1, 2, 3, 4);
    {
        auto astorage =
            grid.template make_storage<icosahedral_topology_t::edges, double, halo<0, 0, 0, 0>, selector<1, 1, 1, 1>>(
                "turu");
        auto ameta = *astorage.get_storage_info_ptr();

        ASSERT_EQ(ameta.total_length<0>(), 4);
        ASSERT_EQ(ameta.total_length<1>(), 3);
        ASSERT_EQ(ameta.total_length<2>(), 6);
        ASSERT_EQ(ameta.total_length<3>(), 7);
#ifdef GT_BACKEND_MC
        // 3rd dimension is padded for MC
        ASSERT_EQ(ameta.padded_length<0>(), 4);
        ASSERT_EQ(ameta.padded_length<1>(), 3);
        ASSERT_EQ(ameta.padded_length<2>(), 6);
        ASSERT_EQ(ameta.padded_length<3>(), 8);
#endif
#ifdef GT_BACKEND_CUDA
        // 3rd dimension is padded for CUDA
        ASSERT_EQ(ameta.padded_length<0>(), 4);
        ASSERT_EQ(ameta.padded_length<1>(), 3);
        ASSERT_EQ(ameta.padded_length<2>(), 6);
        ASSERT_EQ(ameta.padded_length<3>(), 32);
#endif
    }
    {
        auto astorage = grid.template make_storage<icosahedral_topology_t::edges,
            double,
            halo<0, 0, 0, 0, 0, 0>,
            selector<1, 1, 1, 1, 1, 1>>("turu", 8, 9);
        auto ameta = *astorage.get_storage_info_ptr();

        ASSERT_EQ(ameta.total_length<0>(), 4);
        ASSERT_EQ(ameta.total_length<1>(), 3);
        ASSERT_EQ(ameta.total_length<2>(), 6);
        ASSERT_EQ(ameta.total_length<3>(), 7);
#ifdef GT_BACKEND_MC
        // 3rd dimension is padded for MC
        ASSERT_EQ(ameta.padded_length<3>(), 8);
#endif
#ifdef GT_BACKEND_CUDA
        ASSERT_EQ(ameta.padded_length<3>(), 32);
#endif
        ASSERT_EQ(ameta.total_length<4>(), 8);
        ASSERT_EQ(ameta.total_length<5>(), 9);
    }
}
