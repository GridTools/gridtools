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
#pragma once

#include <stencil-composition/stencil-composition.hpp>
#include <communication/low-level/proc_grids_3D.hpp>

#include <boundary-conditions/boundary.hpp>

#include <communication/halo_exchange.hpp>

#include <iostream>
#include <fstream>

#include <tools/mpi_unit_test_driver/check_flags.hpp>
#include <tools/mpi_unit_test_driver/mpi_listener.hpp>
#include <tools/mpi_unit_test_driver/device_binding.hpp>

#include "backend_select.hpp"

/** @file
    @brief This file shows an implementation of the "copy" stencil in parallel with boundary conditions*/

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace copy_stencil {
    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor {
        typedef accessor< 0, enumtype::in > in;
        typedef accessor< 1, enumtype::inout > out;
        typedef boost::mpl::vector< in, out > arg_list;
        /* static const auto expression=in(1,0,0)-out(); */

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(out()) = eval(in());
        }
    };

    /** @brief example of boundary conditions with predicate
     */
    struct boundary_conditions {

        template < typename Direction, typename DataField0, typename DataField1 >
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
        array< int, 3 > dimensions{0, 0, 1};
        MPI_Dims_create(PROCS, 2, &dimensions[0]);
        dimensions[2] = 1;

        typedef storage_traits< backend_t::s_backend_id >::storage_info_t< 0, 3 > storage_info_t;
        typedef storage_traits< backend_t::s_backend_id >::data_store_t< float_type, storage_info_t > storage_t;

        typedef gridtools::halo_exchange_dynamic_ut< typename storage_info_t::layout_t,
            gridtools::layout_map< 0, 1, 2 >,
            float_type,
            MPI_3D_process_grid_t< 3 >,
#ifdef __CUDACC__
            gridtools::gcl_gpu,
#else
            gridtools::gcl_cpu,
#endif
            gridtools::version_manual > pattern_type;

        pattern_type he(gridtools::boollist< 3 >(false, false, false), GCL_WORLD, dimensions);
#ifdef VERBOSE
        printf("halo exchange ok\n");
#endif

        /* The nice interface does not compile today (CUDA 6.5) with nvcc (C++11 support not complete yet)*/

        // Definition of placeholders. The order of them reflect the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain
        typedef arg< 0, storage_t > p_in;
        typedef arg< 1, storage_t > p_out;
        // An array of placeholders to be passed to the domain
        // I'm using mpl::vector, but the final API should look slightly simpler
        typedef boost::mpl::vector< p_in, p_out > accessor_list;

        array< ushort_t, 2 > halo{1, 1};

        if (PROCS == 1) // serial execution
            halo[0] = halo[1] = 0;

        // Definition of the actual data fields that are used for input/output
        he.add_halo< 0 >(halo[0], halo[0], halo[0], d1 + halo[0] - 1, d1 + 2 * halo[0]);
        he.add_halo< 0 >(halo[1], halo[1], halo[1], d2 + halo[1] - 1, d2 + 2 * halo[1]);
        he.add_halo< 0 >(0, 0, 0, d3 - 1, d3);

        he.setup(3);

#ifdef VERBOSE
        printf("halo set up\n");
#endif

        auto c_grid = he.comm();
        int pi, pj, pk;
        c_grid.coords(pi, pj, pk);
        assert(pk == 0);

        storage_info_t storage_info(d1 + 2 * halo[0], d2 + 2 * halo[1], d3);

        storage_t in(storage_info,
            [&storage_info, pi, pj, pk](int i, int j, int k) {
                int I = i + storage_info.dim< 0 >() * pi;
                int J = j + storage_info.dim< 1 >() * pj;
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

        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
        // that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in
        // order. (I don't particularly like this)
        gridtools::aggregator_type< accessor_list > domain(in, out);

        auto copy = gridtools::make_computation< backend_t >(domain,
            grid,
            gridtools::make_multistage // mss_descriptor
            (execute< forward >(), gridtools::make_stage< copy_functor >(p_in(), p_out())));
#ifdef VERBOSE
        printf("computation instantiated\n");
#endif

        copy.steady();

#ifdef VERBOSE
        printf("computation steady\n");
#endif

        copy.run();

#ifdef VERBOSE
        printf("computation run\n");
#endif

        copy.sync_all();

#ifdef VERBOSE
        printf("computation finalized\n");
#endif

        gridtools::array< gridtools::halo_descriptor, 3 > halos;
        halos[0] = gridtools::halo_descriptor(halo[0], halo[0], halo[0], d1 + halo[0] - 1, d1 + 2 * halo[0]);
        halos[1] = gridtools::halo_descriptor(halo[1], halo[1], halo[1], d2 + halo[1] - 1, d2 + 2 * halo[1]);
        halos[2] = gridtools::halo_descriptor(0, 0, 0, d3 - 1, d3);

        typename gridtools::boundary< boundary_conditions,
            backend_t::s_backend_id,
            typename gridtools::proc_grid_predicate< decltype(c_grid) > >(
            halos, boundary_conditions(), gridtools::proc_grid_predicate< decltype(c_grid) >(c_grid))
            .apply(in, out);

#ifdef __CUDACC__
        auto inv = make_device_view(in);
        auto outv = make_device_view(out);
#else
        auto inv = make_host_view(in);
        auto outv = make_host_view(out);
#endif
        std::vector< float_type * > vec(2);
        vec[0] = advanced::get_initial_address_of(inv);
        vec[1] = advanced::get_initial_address_of(outv);

        he.pack(vec);

#ifdef VERBOSE
        printf("copy packed \n");
#endif

        he.exchange();

#ifdef VERBOSE
        printf("copy exchanged\n");
#endif
        he.unpack(vec);

#ifdef VERBOSE
        printf("copy unpacked\n");
#endif

        MPI_Barrier(GCL_WORLD);

        out.sync();
        auto v_out_h = make_host_view< access_mode::ReadOnly >(out);

        for (uint_t i = halo[0]; i < d1 - halo[0]; ++i)
            for (uint_t j = halo[1]; j < d2 - halo[1]; ++j)
                for (uint_t k = 1; k < d3; ++k) {
                    int I = i + storage_info.dim< 0 >() * pi;
                    int J = j + storage_info.dim< 1 >() * pj;
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
