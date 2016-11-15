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

#define ASSERT_GRID_EQ(expect, actual)                         \
    ASSERT_EQ(expect.direction_i(), actual.direction_i());     \
    ASSERT_EQ(expect.direction_j(), actual.direction_j());     \
    for (int i = 0; i < expect.value_list.size(); ++i) {       \
        ASSERT_EQ(expect.value_list[i], actual.value_list[i]); \
    }

class structure_grids_grid : public ::testing::Test {
  protected:
    typedef gridtools::interval< gridtools::level< 0, -1 >, gridtools::level< 1, 1 > > axis;

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

    ASSERT_GRID_EQ(grid_orig, grid_copy);
}

#ifdef __CUDACC__
TEST_F(structure_grids_grid, copy_constructable_device_ptrs_differ) {
    gridtools::grid< axis > grid_copy(grid_orig);

    ASSERT_NE(grid_orig.device_pointer(), grid_copy.device_pointer());
}

// TODO write a test to compare sizes on device
#endif

TEST_F(structure_grids_grid, constructor_with_array) {
#ifdef CXX11_ENABLED
    gridtools::array< gridtools::uint_t, 2 > k_levels{(gridtools::uint_t)0, (gridtools::uint_t)(size_k - 1)};
#else
    // TODO we should have the same ctors available in c++03 and c++11 but changing it clashes with c-style
    // initialization in other code
    gridtools::array< gridtools::uint_t, 2 > k_levels((gridtools::uint_t)0, (gridtools::uint_t)(size_k - 1));
#endif
    gridtools::grid< axis > grid_copy(di, dj, k_levels);

    ASSERT_GRID_EQ(grid_orig, grid_copy);
}

#ifdef CXX11_ENABLED
TEST_F(structure_grids_grid, constructor_make_grid) {
    auto grid_copy = gridtools::make_grid(di, dj, gridtools::make_k_axis(size_k));

    ASSERT_GRID_EQ(grid_orig, grid_copy);
}
#endif
