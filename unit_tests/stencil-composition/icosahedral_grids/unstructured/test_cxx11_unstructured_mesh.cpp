/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
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
#include "common/defs.hpp"
#include "stencil-composition/stencil-composition.hpp"
#include "stencil-composition/icosahedral_grids/unstructured_mesh.hpp"

using namespace gridtools;
using namespace enumtype;

TEST(icosahedral_topology, layout) {

    typedef interval< level< 0, -2 >, level< 1, 1 > > axis;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

    using backend_t = BACKEND;
    using umesh_t = unstructured_mesh;

    atlas::Mesh mesh;
    atlas::mesh::MultiBlockConnectivity &cell_to_node = mesh.cells().node_connectivity();

    {
        const int_t vals[6] = {2, 4, 5, 6, 7, 1};
        cell_to_node.add(3, 2, vals);
    }
    atlas::mesh::MultiBlockConnectivity &cell_to_edge = mesh.cells().edge_connectivity();

    {

        const int_t vals2[6] = {3, 5, 6, 7, 8, 2};
        cell_to_edge.add(3, 2, vals2);
    }
    uint_t halo_size = 2;
    uint_t d1 = 10;
    uint_t d2 = 10;
    grid< axis, icosahedral_topology< BACKEND >, umesh_t > grid_(
        array< uint_t, 5 >{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1},
        array< uint_t, 5 >{halo_size, halo_size, halo_size, d1 - halo_size - 1, d1},
        umesh_t(mesh));

    ASSERT_TRUE(grid_.umesh().connectivity(unstructured_mesh::mb_connectivity_type::cell_to_vertex).size() == 6);
    ASSERT_TRUE(grid_.umesh().connectivity(unstructured_mesh::mb_connectivity_type::cell_to_vertex)(0, 1) == 4);
    ASSERT_TRUE(grid_.umesh().connectivity(unstructured_mesh::mb_connectivity_type::cell_to_vertex)(2, 0) == 7);

    ASSERT_TRUE(grid_.umesh().connectivity(unstructured_mesh::mb_connectivity_type::cell_to_edge).size() == 6);
    ASSERT_TRUE(grid_.umesh().connectivity(unstructured_mesh::mb_connectivity_type::cell_to_edge)(1, 1) == 7);
    ASSERT_TRUE(grid_.umesh().connectivity(unstructured_mesh::mb_connectivity_type::cell_to_edge)(2, 1) == 2);

    //    grid_.clone_to_device();
}
