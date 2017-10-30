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

#include <gridtools.hpp>
#include <stencil-composition/stencil-composition.hpp>
#include "gtest/gtest.h"

namespace check_grid_bounds {

    typedef gridtools::backend< gridtools::enumtype::Host,
        gridtools::enumtype::GRIDBACKEND,
        gridtools::enumtype::Block > the_backend;

    typedef gridtools::storage_traits< gridtools::enumtype::Host > storage_tr;

    /**
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
    bool test(gridtools::uint_t x,
        gridtools::uint_t y,
        gridtools::uint_t z,
        gridtools::uint_t gx,
        gridtools::uint_t gy,
        gridtools::uint_t gz,
        bool expected) {

        using storage_info1_t = storage_tr::storage_info_t< 0, 3, gridtools::halo< 2, 2, 0 > >;
        using storage_info2_t = storage_tr::storage_info_t< 1, 3, gridtools::halo< 2, 2, 0 > >;
        using storage_info3_t = storage_tr::storage_info_t< 2, 3, gridtools::halo< 2, 2, 0 > >;

        using storage_type1 = storage_tr::data_store_t< double, storage_info1_t >;
        using storage_type2 = storage_tr::data_store_t< int, storage_info2_t >;
        using storage_type3 = storage_tr::data_store_t< float, storage_info3_t >;

        // TODO: Use storage_info as unnamed object - lifetime issues on GPUs
        storage_info1_t si1(x + 3, y, z);
        storage_info2_t si2(x, y + 2, z);
        storage_info3_t si3(x, y, z + 1);

        storage_type1 field1 = storage_type1(si1, double(), "field1");
        storage_type2 field2 = storage_type2(si2, int(), "field2");
        storage_type3 field3 = storage_type3(si3, float(), "field3");

        typedef gridtools::arg< 0, storage_type1 > p_field1;
        typedef gridtools::arg< 1, storage_type2 > p_field2;
        typedef gridtools::arg< 2, storage_type3 > p_field3;

        typedef boost::mpl::vector< p_field1, p_field2, p_field3 > accessor_list;

        gridtools::aggregator_type< accessor_list > domain(
            (p_field1() = field1), (p_field2() = field2), (p_field3() = field3));

        gridtools::uint_t halo_size = 0;
        gridtools::halo_descriptor di{halo_size, halo_size, halo_size, gx - halo_size - 1, gx};
        gridtools::halo_descriptor dj{halo_size, halo_size, halo_size, gy - halo_size - 1, gy};

        auto grid = make_grid(di, dj, gridtools::axis< 1 >(gz));

        using mdlist_t = boost::fusion::vector< storage_info1_t, storage_info2_t, storage_info3_t >;
        mdlist_t mdlist(si1, si2, si3);

        try {
            gridtools::check_fields_sizes< gridtools::grid_traits_from_id< gridtools::enumtype::structured > >(
                grid, domain);
        } catch (std::runtime_error const &err) {
            expected = !expected;
        }

        return expected;
    }
} // namespace check_grid_bounds

// Tests that are assumed to pass, so that the last argument is `true`, which is them returned as is
TEST(stencil_composition, check_grid_bounds1) { EXPECT_TRUE(check_grid_bounds::test(4, 5, 6, 4, 5, 6, true)); }
TEST(stencil_composition, check_grid_bounds2) { EXPECT_TRUE(check_grid_bounds::test(4, 5, 6, 4, 5, 6, true)); }

// Tests that are assumed to fail, so that the last argument is `false`, which is them returned flipped when exception
// is catched
TEST(stencil_composition, check_grid_bounds3) { EXPECT_TRUE(check_grid_bounds::test(4, 5, 6, 8, 5, 7, false)); }
TEST(stencil_composition, check_grid_bounds4) { EXPECT_TRUE(check_grid_bounds::test(4, 5, 6, 4, 5, 7, false)); }
TEST(stencil_composition, check_grid_bounds5) { EXPECT_TRUE(check_grid_bounds::test(4, 5, 6, 4, 12, 6, false)); }
TEST(stencil_composition, check_grid_bounds6) { EXPECT_TRUE(check_grid_bounds::test(4, 5, 6, 9, 5, 6, false)); }
