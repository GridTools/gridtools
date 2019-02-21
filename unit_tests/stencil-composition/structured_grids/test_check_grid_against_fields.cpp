/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil-composition/intermediate.hpp>

#include <gtest/gtest.h>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/halo_descriptor.hpp>
#include <gridtools/stencil-composition/backend_ids.hpp>
#include <gridtools/storage/common/halo.hpp>
#include <gridtools/storage/storage-facility.hpp>

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

        using storage_info_t = storage_traits<target::x86>::storage_info_t<0, 3, halo<2, 2, 0>>;

        halo_descriptor di = {0, 0, 0, gx - 1, gx};
        halo_descriptor dj = {0, 0, 0, gy - 1, gy};

        auto grid = make_grid(di, dj, gz);
        auto testee = storage_info_fits_grid<backend_ids<target::x86, strategy::block>>(grid);

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
} // namespace gridtools
