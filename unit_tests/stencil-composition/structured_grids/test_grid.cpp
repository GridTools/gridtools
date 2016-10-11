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

#include <common/defs.hpp>
#include <stencil-composition/structured_grids/grid.hpp>
#include <boost/type_index.hpp>

class structure_grids_grid : public ::testing::Test {
  protected:
    typedef gridtools::interval< gridtools::level< 0, -2 >, gridtools::level< 1, 1 > > axis;

    static const gridtools::uint_t size_i = 22;
    static const gridtools::uint_t size_j = 33;
    static const gridtools::uint_t size_k = 44;

    gridtools::halo_descriptor di;
    gridtools::halo_descriptor dj;

    gridtools::grid< axis > grid_orig;

    structure_grids_grid() : di(0, 0, 0, size_i - 1, size_i), dj(0, 0, 0, size_j - 1, size_j), grid_orig(di, dj) {
        grid_orig.value_list[0] = 0;
        grid_orig.value_list[1] = size_k - 1;
    }
};

TEST_F(structure_grids_grid, copy_constructable) {
    gridtools::grid< axis > grid_copy(grid_orig);

    ASSERT_EQ(grid_orig.direction_i(), grid_copy.direction_i());
    ASSERT_EQ(grid_orig.direction_j(), grid_copy.direction_j());

    for (int i = 0; i < grid_orig.value_list.size(); ++i) {
        ASSERT_EQ(grid_orig.value_list[i], grid_copy.value_list[i]);
    }
}

#ifdef __CUDACC__
TEST_F(structure_grids_grid, copy_constructable_device_ptrs_differ) {
    gridtools::grid< axis > grid_copy(grid_orig);

    ASSERT_NE(grid_orig.device_pointer(), grid_copy.device_pointer());
}

// TODO compare sizes on device
#endif

TEST_F(structure_grids_grid, copy_constructor_with_array) {
    gridtools::array< gridtools::uint_t, 2 > k_levels((gridtools::uint_t)0, (gridtools::uint_t)(size_k - 1));
    gridtools::grid< axis > grid_copy(di, dj, k_levels);

    ASSERT_EQ(grid_orig.direction_i(), grid_copy.direction_i());
    ASSERT_EQ(grid_orig.direction_j(), grid_copy.direction_j());

    for (int i = 0; i < grid_orig.value_list.size(); ++i) {
        ASSERT_EQ(grid_orig.value_list[i], grid_copy.value_list[i]);
    }
}

#ifdef CXX11_ENABLED
TEST_F(structure_grids_grid, copy_constructor_make_k_levels) {
    gridtools::grid< axis > grid_copy(
        di, dj, gridtools::make_k_levels((gridtools::uint_t)0, (gridtools::uint_t)(size_k - 1)));

    ASSERT_EQ(grid_orig.direction_i(), grid_copy.direction_i());
    ASSERT_EQ(grid_orig.direction_j(), grid_copy.direction_j());

    for (int i = 0; i < grid_orig.value_list.size(); ++i) {
        ASSERT_EQ(grid_orig.value_list[i], grid_copy.value_list[i]);
    }
}
#endif
