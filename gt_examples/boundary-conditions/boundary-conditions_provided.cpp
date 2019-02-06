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

/** @file This file contains several examples of using boundary
    conditions classes provided by gridtools itself.

    They are:

    - copy_boundary, that takes 2 or 3 fields, and copy the values at
      the boundary of the last one into the others;

    - zero_boundary, that set the boundary of the fields (maximum 3 in
      current implementation) to the default constructed value of the
      data_store value type;

    - value_boundary, that set the boundary to a specified value.

    We are using helper functions to show how to use them and a simple
    code to check correctness.
 */

#include <iostream>

#include <gridtools/boundary-conditions/boundary.hpp>
#include <gridtools/tools/backend_select.hpp>

#include <gridtools/boundary-conditions/copy.hpp>
#include <gridtools/boundary-conditions/value.hpp>
#include <gridtools/boundary-conditions/zero.hpp>

namespace gt = gridtools;

#ifdef __CUDACC__
using target_t = gt::target::cuda;
using strategy_t = gt::strategy::block;
#else
using target_t = gt::target::mc;
using strategy_t = gt::strategy::block;
#endif

using backend_t = gt::backend<target_t, gt::grid_type::structured, strategy_t>;

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0]
                  << " dimx dimy dimz\n"
                     " where args are integer sizes of the data fields"
                  << std::endl;
        return EXIT_FAILURE;
    }

    using uint_t = unsigned;

    uint_t d1 = atoi(argv[1]);
    uint_t d2 = atoi(argv[2]);
    uint_t d3 = atoi(argv[3]);

    using storage_info_t = backend_t::storage_traits_t::storage_info_t<0, 3, gt::halo<1, 1, 1>>;
    using storage_t = backend_t::storage_traits_t::data_store_t<int, storage_info_t>;

    // Definition of the actual data fields that are used for input/output
    storage_info_t storage_info(d1, d2, d3);
    storage_t in_s(storage_info, [](int i, int j, int k) { return i + j + k; }, "in");
    storage_t out_s(storage_info, 0, "out");

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
    gt::array<gt::halo_descriptor, 3> halos;
    halos[0] = gt::halo_descriptor(1, 1, 1, d1 - 2, d1);
    halos[1] = gt::halo_descriptor(1, 1, 1, d2 - 2, d2);
    halos[2] = gt::halo_descriptor(1, 1, 1, d3 - 2, d3);

    bool error = false;
    {
        // sync the data stores if needed
        in_s.sync();
        out_s.sync();

        gt::boundary<gt::copy_boundary, backend_t::backend_id_t>(halos, gt::copy_boundary{}).apply(out_s, in_s);

        // sync the data stores if needed
        in_s.sync();
        out_s.sync();

        // making the views to access and check correctness
        auto in = make_host_view(in_s);
        auto out = make_host_view(out_s);

        assert(check_consistency(in_s, in) && "view is in an inconsistent state.");
        assert(check_consistency(out_s, out) && "view is in an inconsistent state.");

        // reactivate views and check consistency
        out_s.reactivate_host_write_views();
        assert(check_consistency(in_s, in) && "view is in an inconsistent state.");
        assert(check_consistency(out_s, out) && "view is in an inconsistent state.");

        for (uint_t i = 0; i < d1; ++i) {
            for (uint_t j = 0; j < d2; ++j) {
                for (uint_t k = 0; k < d3; ++k) {
                    // check outer surfaces of the cube
                    if ((i == 0 || i == d1 - 1) || (j == 0 || j == d2 - 1) || (k == 0 || k == d3 - 1)) {
                        error |= out(i, j, k) != i + j + k;
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

        gt::boundary<gt::zero_boundary, backend_t::backend_id_t>(halos, gt::zero_boundary{}).apply(out_s);

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

        gt::boundary<gt::value_boundary<int>, backend_t::backend_id_t>(halos, gt::value_boundary<int>{42}).apply(out_s);

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
