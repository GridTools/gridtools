/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#include "gtest/gtest.h"

#include <common/defs.hpp>
#include <common/gt_assert.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include <stencil-composition/icosahedral_grids/icosahedral_topology.hpp>
#include "backend_select.hpp"

using namespace gridtools;

using icosahedral_topology_t = icosahedral_topology< backend_t >;
TEST(icosahedral_topology, layout) {
    using alayout_t = icosahedral_topology_t::layout_t< selector< 1, 1, 1, 1 > >;
#ifdef __CUDACC__
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< alayout_t, layout_map< 3, 2, 1, 0 > >::value), "ERROR");
#else
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< alayout_t, layout_map< 0, 1, 2, 3 > >::value), "ERROR");
#endif

    using alayout_2d_t = icosahedral_topology_t::layout_t< selector< 1, 1, 1, 0 > >;
#ifdef __CUDACC__
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< alayout_2d_t, layout_map< 2, 1, 0, -1 > >::value), "ERROR");
#else
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< alayout_2d_t, layout_map< 0, 1, 2, -1 > >::value), "ERROR");
#endif

    using alayout_6d_t = icosahedral_topology_t::layout_t< selector< 1, 1, 1, 1, 1, 1 > >;
#ifdef __CUDACC__
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< alayout_6d_t, layout_map< 5, 4, 3, 2, 1, 0 > >::value), "ERROR");
#else
    GRIDTOOLS_STATIC_ASSERT((boost::is_same< alayout_6d_t, layout_map< 2, 3, 4, 5, 0, 1 > >::value), "ERROR");
#endif
}

TEST(icosahedral_topology, make_storage) {

    icosahedral_topology_t grid(4, 6, 7);
    icosahedral_topology_t::meta_storage_t< icosahedral_topology_t::edges, halo< 0, 0, 0, 0 >, selector< 1, 1, 1, 1 > >
        x(1, 2, 3, 4);
    {
        auto astorage = grid.template make_storage< icosahedral_topology_t::edges,
            double,
            halo< 0, 0, 0, 0 >,
            selector< 1, 1, 1, 1 > >("turu");
        auto ameta = *astorage.get_storage_info_ptr();

        ASSERT_TRUE((ameta.dim< 0 >() == 4));
        ASSERT_TRUE((ameta.dim< 1 >() == 3));
        ASSERT_TRUE((ameta.dim< 2 >() == 6));
        ASSERT_TRUE((ameta.dim< 3 >() == 7));
#ifdef BACKEND_MIC
        fuck
        // 3rd dimension is padded for MIC
        ASSERT_EQ(ameta.total_length< 3 >(), 8);
#endif
    }
    {
        auto astorage = grid.template make_storage< icosahedral_topology_t::edges,
            double,
            halo< 0, 0, 0, 0, 0, 0 >,
            selector< 1, 1, 1, 1, 1, 1 > >("turu", 8, 9);
        auto ameta = *astorage.get_storage_info_ptr();

        ASSERT_TRUE((ameta.dim< 0 >() == 4));
        ASSERT_TRUE((ameta.dim< 1 >() == 3));
#ifdef BACKEND_MIC
        // 3rd dimension is padded for MIC
        ASSERT_TRUE((ameta.dim< 3 >() == 8));
#else
        ASSERT_TRUE((ameta.dim< 3 >() == 7));
#endif
        ASSERT_TRUE((ameta.dim< 4 >() == 8));
        ASSERT_TRUE((ameta.dim< 5 >() == 9));
    }
}

TEST(icosahedral_topology, make_tmp_storage) {}
