/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <fstream>
#include <iostream>

#include <gridtools/boundary_conditions/boundary.hpp>
#include <gridtools/communication/halo_exchange.hpp>
#include <gridtools/communication/low_level/proc_grids_3D.hpp>
#include <gridtools/distributed_boundaries/grid_predicate.hpp>
#include <gridtools/stencil_composition/cartesian.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>
#include <gridtools/storage/traits.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <gridtools/tools/mpi_unit_test_driver/check_flags.hpp>
#include <gridtools/tools/mpi_unit_test_driver/device_binding.hpp>
#include <gridtools/tools/mpi_unit_test_driver/mpi_listener.hpp>

/** @file
    @brief This file shows an implementation of the "copy" stencil in parallel with boundary conditions*/

using namespace gridtools;
using namespace cartesian;

namespace copy_stencil {
    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor {
        typedef accessor<0, intent::in> in;
        typedef accessor<1, intent::inout> out;
        typedef make_param_list<in, out> param_list;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

    /** @brief example of boundary conditions with predicate
     */
    struct boundary_conditions {

        template <typename Direction, typename DataField0, typename DataField1>
        GT_FUNCTION void operator()(
            Direction, DataField0 &data_field0, DataField1 &data_field1, uint_t i, uint_t j, uint_t k) const {
            data_field0(i, j, k) = -1.11111111;
            data_field1(i, j, k) = -1.11111111;
        }
    };

    bool test(uint_t d1, uint_t d2, uint_t d3) {

        //! [proc_grid_dims]
        MPI_Comm CartComm;
        array<int, 3> dimensions{0, 0, 1};
        int period[3] = {1, 1, 1};
        MPI_Dims_create(PROCS, 2, &dimensions[0]);
        assert(dimensions[2] == 1);

        MPI_Cart_create(MPI_COMM_WORLD, 3, &dimensions[0], period, false, &CartComm);

        using pattern_type = halo_exchange_dynamic_ut<storage::traits::layout_type<storage_traits_t, 3>,
            layout_map<0, 1, 2>,
            float_type,
            gcl_arch_t>;

        pattern_type he(boollist<3>(false, false, false), CartComm);

        array<uint_t, 2> halo{1, 1};

        if (PROCS == 1) // serial execution
            halo[0] = halo[1] = 0;

        // Definition of the actual data fields that are used for input/output
        he.add_halo<0>(halo[0], halo[0], halo[0], d1 + halo[0] - 1, d1 + 2 * halo[0]);
        he.add_halo<0>(halo[1], halo[1], halo[1], d2 + halo[1] - 1, d2 + 2 * halo[1]);
        he.add_halo<0>(0, 0, 0, d3 - 1, d3);

        he.setup(3);

        auto c_grid = he.comm();
        int pi, pj, pk;
        c_grid.coords(pi, pj, pk);
        assert(pk == 0);

        size_t x = d1 + 2 * halo[0];
        size_t y = d2 + 2 * halo[1];

        auto builder = storage::builder<storage_traits_t>.type<float_type>().dimensions(x, y, d3);

        auto input = [&](int i, int j, int k) {
            int I = i + x * pi;
            int J = j + y * pj;
            int K = k;
            return I + J + K;
        };

        auto in = builder.initializer(input)();
        auto out = builder.value(-2.2222222)();

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        auto grid = make_grid({halo[0], halo[0], halo[0], d1 + halo[0] - 1, d1 + 2 * halo[0]},
            {halo[1], halo[1], halo[1], d2 + halo[1] - 1, d2 + 2 * halo[1]},
            d3);

        easy_run(copy_functor(), backend_t(), grid, in, out);

        array<halo_descriptor, 3> halos;
        halos[0] = halo_descriptor(halo[0], halo[0], halo[0], d1 + halo[0] - 1, d1 + 2 * halo[0]);
        halos[1] = halo_descriptor(halo[1], halo[1], halo[1], d2 + halo[1] - 1, d2 + 2 * halo[1]);
        halos[2] = halo_descriptor(0, 0, 0, d3 - 1, d3);

        boundary<boundary_conditions, backend_t, proc_grid_predicate<decltype(c_grid)>>(
            halos, boundary_conditions(), proc_grid_predicate<decltype(c_grid)>(c_grid))
            .apply(in, out);

        std::vector<float_type *> vec = {in->get_target_ptr(), out->get_target_ptr()};

        he.pack(vec);
        he.exchange();
        he.unpack(vec);

        MPI_Barrier(GCL_WORLD);

        auto v_out_h = out->const_host_view();

        for (uint_t i = halo[0]; i < d1 - halo[0]; ++i)
            for (uint_t j = halo[1]; j < d2 - halo[1]; ++j)
                for (uint_t k = 1; k < d3; ++k) {
                    if (v_out_h(i, j, k) != input(i, j, k)) {
                        std::cout << PID << " "
                                  << "i = " << i << ", j = " << j << ", k = " << k
                                  << "v_out_h(i, j, k) = " << v_out_h(i, j, k) << ", "
                                  << "(I + J + K) = " << (i + j + k) * (PID + 1) << "\n";
                        return false;
                    }
                }

        std::cout << "(" << PID << ") Completed\n";

        return true;
    }

} // namespace copy_stencil
