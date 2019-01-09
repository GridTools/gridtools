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

/** @file
    This file contains several examples of using boundary conditions

    The basic concept, here and in distributed boundaries and
    communication is the concept of "direction".

    In a 3D regular grid, which is where this implementation of the
    boundary condition library applies, we associate a 3D axis system,
    and the cell indices (i,j,k) naturally lie on it. With this axis
    system the concept of "vector" can be defined to indicate
    distances and directions. Direction is the one think we need
    here. Instead of using unitary vectors to indicate directions, as
    it is usually the case for euclidean spaces, we use vectors whose
    components are -1, 0, and 1.  For example, (1, 1, 1) is the
    dicretion indicated by the unit vector (1,1,1)/sqrt(3).

    If we take the center of a 3D grid, then we can define 26
    different directions {(i,j,k): i,j,k \in {-1, 0, 1}}\{0,0,0} that
    identify the different faces, edges and corners of the cube to
    which the grid is topologically analogous with.

    THE MAIN IDEA:
    A boundary condition class specialize operator() to accept a
    direction and when that diretction is accessed, the data fields in
    the boundary corresponding to that direction can be accessed.
 */

#include <gridtools/boundary-conditions/boundary.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <iostream>

#include <gridtools/boundary-conditions/copy.hpp>
#include <gridtools/boundary-conditions/value.hpp>
#include <gridtools/boundary-conditions/zero.hpp>

using namespace gridtools;
using namespace enumtype;

template <typename Halo, typename Field1, typename Field2>
void apply_copy(Halo const &halos, Field1 &field1, Field2 const &field2) {
    gridtools::template boundary<copy_boundary, backend_t::backend_id_t>(halos, copy_boundary{}).apply(field1, field2);
}

template <typename Halo, typename Field1>
void apply_zero(Halo const &halos, Field1 &field1) {
    gridtools::template boundary<zero_boundary, backend_t::backend_id_t>(halos, zero_boundary{}).apply(field1);
}

template <typename Halo, typename Field1, typename T>
void apply_value(Halo const &halos, Field1 &field1, T value) {
    gridtools::template boundary<value_boundary<T>, backend_t::backend_id_t>(halos, value_boundary<T>{value})
        .apply(field1);
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0]
                  << " dimx dimy dimz\n"
                     " where args are integer sizes of the data fields"
                  << std::endl;
        return EXIT_FAILURE;
    }

    uint_t d1 = atoi(argv[1]);
    uint_t d2 = atoi(argv[2]);
    uint_t d3 = atoi(argv[3]);

    typedef backend_t::storage_traits_t::storage_info_t<0, 3, halo<1, 1, 1>> meta_data_t;
    typedef backend_t::storage_traits_t::data_store_t<int_t, meta_data_t> storage_t;

    // Definition of the actual data fields that are used for input/output
    meta_data_t meta_(d1, d2, d3);
    storage_t in_s(meta_, [](int i, int j, int k) { return i + j + k; }, "in");
    storage_t out_s(meta_, 0, "out");

    /* Defintion of the boundaries of the storage. We use
       halo_descriptor, that are used also in the current
       communication library. We plan to use a better structure in the
       future. The halo descriptor contains 5 numbers:
       - The halo in the minus direction
       - The halo in the plus direction
       - The begin of the inner region
       - The end (inclusive) of the inner region
       - The total length if the dimension.

       You need 3 halo descriptors, one per dimension.
    */
    gridtools::array<gridtools::halo_descriptor, 3> halos;
    halos[0] = gridtools::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gridtools::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gridtools::halo_descriptor(1, 1, 1, d3 - 2, d3);

    bool error = false;
    {
        // sync the data stores if needed
        in_s.sync();
        out_s.sync();

        apply_copy(halos, out_s, in_s);

        // sync the data stores if needed
        in_s.sync();
        out_s.sync();

        // making the views to access and check correctness
        auto in = make_host_view(in_s);
        auto out = make_host_view(out_s);

        assert(check_consistency(in_s, in) && "view is in an inconsistent state.");
        assert(check_consistency(out_s, out) && "view is in an inconsistent state.");

        // reactivate views and check consistency
        in_s.reactivate_host_write_views();
        out_s.reactivate_host_write_views();
        assert(check_consistency(in_s, in) && "view is in an inconsistent state.");
        assert(check_consistency(out_s, out) && "view is in an inconsistent state.");

        for (uint_t i = 0; i < d1; ++i) {
            for (uint_t j = 0; j < d2; ++j) {
                for (uint_t k = 0; k < d3; ++k) {
                    // check outer surfaces of the cube
                    if ((i == 0 || i == d1 - 1) || (j == 0 || j == d2 - 1) || (k == 0 || k == d3 - 1)) {
                        error |= out(i, j, k) != i + j + k;
                    } else {
                        error |= out(i, j, k) != 0;
                        error |= in(i, j, k) != i + j + k;
                    }
                }
            }
        }
    }

    {
        // sync the data stores if needed
        out_s.sync();

        apply_zero(halos, out_s);

        // sync the data stores if needed
        out_s.sync();

        // making the views to access and check correctness
        auto out = make_host_view(out_s);

        assert(check_consistency(out_s, out) && "view is in an inconsistent state.");

        // reactivate views and check consistency
        out_s.reactivate_host_write_views();
        assert(check_consistency(out_s, out) && "view is in an inconsistent state.");

        for (uint_t i = 0; i < d1; ++i) {
            for (uint_t j = 0; j < d2; ++j) {
                for (uint_t k = 0; k < d3; ++k) {
                    // check outer surfaces of the cube
                    if ((i == 0 || i == d1 - 1) || (j == 0 || j == d2 - 1) || (k == 0 || k == d3 - 1)) {
                        error |= out(i, j, k) != 0;
                    } else {
                        error |= out(i, j, k) != 0;
                    }
                }
            }
        }
    }

    {
        // sync the data stores if needed
        out_s.sync();

        apply_value(halos, out_s, 42);

        // sync the data stores if needed
        out_s.sync();

        // making the views to access and check correctness
        auto out = make_host_view(out_s);

        assert(check_consistency(out_s, out) && "view is in an inconsistent state.");

        // reactivate views and check consistency
        out_s.reactivate_host_write_views();
        assert(check_consistency(out_s, out) && "view is in an inconsistent state.");

        for (uint_t i = 0; i < d1; ++i) {
            for (uint_t j = 0; j < d2; ++j) {
                for (uint_t k = 0; k < d3; ++k) {
                    // check outer surfaces of the cube
                    if ((i == 0 || i == d1 - 1) || (j == 0 || j == d2 - 1) || (k == 0 || k == d3 - 1)) {
                        error |= out(i, j, k) != 42;
                    } else {
                        error |= out(i, j, k) != 0;
                    }
                }
            }
        }
    }

    if (error) {
        std::cout << "TEST failed.\n";
    } else {
        std::cout << "TEST passed.\n";
    }

    return error;
}
