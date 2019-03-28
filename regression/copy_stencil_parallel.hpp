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
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/backend_select.hpp>
#include <gridtools/tools/mpi_unit_test_driver/check_flags.hpp>
#include <gridtools/tools/mpi_unit_test_driver/device_binding.hpp>
#include <gridtools/tools/mpi_unit_test_driver/mpi_listener.hpp>

/** @file
    @brief This file shows an implementation of the "copy" stencil in parallel with boundary conditions*/

using gridtools::accessor;
using gridtools::arg;
using gridtools::extent;
using gridtools::level;

using namespace gridtools;
using namespace execute;

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

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, copy_functor const) { return s << "copy_functor"; }

    bool test(uint_t d1, uint_t d2, uint_t d3) {

        //! [proc_grid_dims]
        MPI_Comm CartComm;
        array<int, 3> dimensions{0, 0, 1};
        int period[3] = {1, 1, 1};
        MPI_Dims_create(PROCS, 2, &dimensions[0]);
        assert(dimensions[2] == 1);

        MPI_Cart_create(MPI_COMM_WORLD, 3, &dimensions[0], period, false, &CartComm);

        typedef storage_traits<backend_t>::storage_info_t<0, 3> storage_info_t;
        typedef storage_traits<backend_t>::data_store_t<float_type, storage_info_t> storage_t;

        typedef gridtools::halo_exchange_dynamic_ut<typename storage_info_t::layout_t,
            gridtools::layout_map<0, 1, 2>,
            float_type,
            MPI_3D_process_grid_t<3>,
#ifdef __CUDACC__
            gridtools::gcl_gpu,
#else
            gridtools::gcl_cpu,
#endif
            gridtools::version_manual>
            pattern_type;

        pattern_type he(gridtools::boollist<3>(false, false, false), CartComm);
#ifdef GT_VERBOSE
        std::cout << "halo exchange ok" << std::endl;
#endif

        /* The nice interface does not compile today (CUDA 6.5) with nvcc (C++11 support not complete yet)*/

        // Definition of placeholders. The order of them reflect the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain
        typedef arg<0, storage_t> p_in;
        typedef arg<1, storage_t> p_out;

        array<ushort_t, 2> halo{1, 1};

        if (PROCS == 1) // serial execution
            halo[0] = halo[1] = 0;

        // Definition of the actual data fields that are used for input/output
        he.add_halo<0>(halo[0], halo[0], halo[0], d1 + halo[0] - 1, d1 + 2 * halo[0]);
        he.add_halo<0>(halo[1], halo[1], halo[1], d2 + halo[1] - 1, d2 + 2 * halo[1]);
        he.add_halo<0>(0, 0, 0, d3 - 1, d3);

        he.setup(3);

#ifdef GT_VERBOSE
        std::cout << "halo set up" << std::endl;
#endif

        auto c_grid = he.comm();
        int pi, pj, pk;
        c_grid.coords(pi, pj, pk);
        assert(pk == 0);

        storage_info_t storage_info(d1 + 2 * halo[0], d2 + 2 * halo[1], d3);

        storage_t in(storage_info,
            [&storage_info, pi, pj](int i, int j, int k) {
                int I = i + storage_info.total_length<0>() * pi;
                int J = j + storage_info.total_length<1>() * pj;
                int K = k;
                return I + J + K;
            },
            "in");
        storage_t out(storage_info, -2.2222222, "out");

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        auto grid = make_grid({halo[0], halo[0], halo[0], d1 + halo[0] - 1, d1 + 2 * halo[0]},
            {halo[1], halo[1], halo[1], d2 + halo[1] - 1, d2 + 2 * halo[1]},
            d3);

        auto copy = gridtools::make_computation<backend_t>(grid,
            p_in{} = in,
            p_out{} = out,
            gridtools::make_multistage // mss_descriptor
            (execute::forward(), gridtools::make_stage<copy_functor>(p_in(), p_out())));
#ifdef GT_VERBOSE
        std::cout << "computation instantiated" << std::endl;
#endif

#ifdef GT_VERBOSE
        std::cout << "computation steady" << std::endl;
#endif

        copy.run();

#ifdef GT_VERBOSE
        std::cout << "computation run" << std::endl;
#endif

#ifdef GT_VERBOSE
        std::cout << "computation finalized" << std::endl;
#endif

        gridtools::array<gridtools::halo_descriptor, 3> halos;
        halos[0] = gridtools::halo_descriptor(halo[0], halo[0], halo[0], d1 + halo[0] - 1, d1 + 2 * halo[0]);
        halos[1] = gridtools::halo_descriptor(halo[1], halo[1], halo[1], d2 + halo[1] - 1, d2 + 2 * halo[1]);
        halos[2] = gridtools::halo_descriptor(0, 0, 0, d3 - 1, d3);

        typename gridtools::
            boundary<boundary_conditions, backend_t, typename gridtools::proc_grid_predicate<decltype(c_grid)>>(
                halos, boundary_conditions(), gridtools::proc_grid_predicate<decltype(c_grid)>(c_grid))
                .apply(in, out);

#ifdef __CUDACC__
        auto inv = make_device_view(in);
        auto outv = make_device_view(out);
#else
        auto inv = make_host_view(in);
        auto outv = make_host_view(out);
#endif
        std::vector<float_type *> vec(2);
        vec[0] = advanced::get_raw_pointer_of(inv);
        vec[1] = advanced::get_raw_pointer_of(outv);

        he.pack(vec);

#ifdef GT_VERBOSE
        std::cout << "copy packed " << std::endl;
#endif

        he.exchange();

#ifdef GT_VERBOSE
        std::cout << "copy exchanged" << std::endl;
#endif
        he.unpack(vec);

#ifdef GT_VERBOSE
        std::cout << "copy unpacked" << std::endl;
#endif

        MPI_Barrier(GCL_WORLD);

        out.sync();
        auto v_out_h = make_host_view<access_mode::read_only>(out);

        for (uint_t i = halo[0]; i < d1 - halo[0]; ++i)
            for (uint_t j = halo[1]; j < d2 - halo[1]; ++j)
                for (uint_t k = 1; k < d3; ++k) {
                    int I = i + storage_info.total_length<0>() * pi;
                    int J = j + storage_info.total_length<1>() * pj;
                    int K = k;

                    if (v_out_h(i, j, k) != (I + J + K)) {
                        std::cout << gridtools::PID << " "
                                  << "i = " << i << ", j = " << j << ", k = " << k
                                  << "v_out_h(i, j, k) = " << v_out_h(i, j, k) << ", "
                                  << "(I + J + K) = " << (i + j + k) * (gridtools::PID + 1) << "\n";
                        return false;
                    }
                }

        std::cout << "(" << gridtools::PID << ") Completed\n";

        return true;
    }

} // namespace copy_stencil
