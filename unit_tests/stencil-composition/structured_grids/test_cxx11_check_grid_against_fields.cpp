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

#include <stencil-composition/intermediate.hpp>

#include <gtest/gtest.h>

#include <common/defs.hpp>
#include <common/halo_descriptor.hpp>
#include <stencil-composition/grid_traits.hpp>
#include <storage/common/halo.hpp>
#include <storage/storage-facility.hpp>

namespace gridtools {

    /*
       Testing the check of bounds for iteration spaces.

       The first three arguments are the sizes of the storages, the
       second three are the sizes of the grid.

       If the sizes would be ok for a computation to run, then the
       check will not throw an exception. In this case the last
       argument will be returned to the caller. If the sizes make the
       check to fail, then the returned value if the negation of the
       last argument. This is why, the test is given a `true` if the
       test is assumed to pass, and a `false` if it assumed to
       fail. In this way we can test both cases, the failing and the
       successful.
     */
    bool do_test(uint_t x, uint_t y, uint_t z, uint_t gx, uint_t gy, uint_t gz) {

        using storage_info_t = storage_traits< enumtype::Host >::storage_info_t< 0, 3, halo< 2, 2, 0 > >;

        halo_descriptor di = {0, 0, 0, gx - 1, gx};
        halo_descriptor dj = {0, 0, 0, gy - 1, gy};

        auto grid = make_grid(di, dj, gz);
        auto testee = storage_info_fits_grid< grid_traits_from_id< enumtype::structured > >(grid);

        return testee(storage_info_t{x + 3, y, z}) && testee(storage_info_t{x, y + 2, z}) &&
               testee(storage_info_t{x, y, z + 1});
    }

    // Tests that are assumed to pass
    TEST(stencil_composition, check_grid_bounds1) { EXPECT_TRUE(do_test(4, 5, 6, 4, 5, 6)); }

    // Tests that are assumed to fail
    TEST(stencil_composition, check_grid_bounds3) { EXPECT_FALSE(do_test(4, 5, 6, 8, 5, 7)); }
    TEST(stencil_composition, check_grid_bounds4) { EXPECT_FALSE(do_test(4, 5, 6, 4, 5, 7)); }
    TEST(stencil_composition, check_grid_bounds5) { EXPECT_FALSE(do_test(4, 5, 6, 4, 12, 6)); }
    TEST(stencil_composition, check_grid_bounds6) { EXPECT_FALSE(do_test(4, 5, 6, 9, 5, 6)); }
}
