/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
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

#include <gridtools/boundary_conditions/boundary.hpp>
#include <gridtools/tools/backend_select.hpp>

#include <gridtools/boundary_conditions/copy.hpp>
#include <gridtools/boundary_conditions/value.hpp>
#include <gridtools/boundary_conditions/zero.hpp>

namespace gt = gridtools;

#ifdef __CUDACC__
using target_t = gt::target::cuda;
#else
using target_t = gt::target::mc;
#endif

using backend_t = gt::backend<target_t>;

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
