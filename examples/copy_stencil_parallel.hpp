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
#pragma once

#include <stencil-composition/stencil-composition.hpp>
#include <storage/partitioner_trivial.hpp>
#include <storage/parallel_storage.hpp>
#include <stencil-composition/interval.hpp>
#include <stencil-composition/make_computation.hpp>
#include <communication/low-level/proc_grids_3D.hpp>

#include <communication/halo_exchange.hpp>

/** @file
    @brief This file shows an implementation of the "copy" stencil in parallel, simple copy of one field done on the
   backend*/

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace copy_stencil {
    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor {
        typedef accessor< 0, enumtype::in > in;
        typedef accessor< 1, enumtype::inout > out;
        typedef boost::mpl::vector< in, out > arg_list;
        /* static const auto expression=in(1,0,0)-out(); */

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, copy_functor const) { return s << "copy_functor"; }

    bool test(uint_t d1, uint_t d2, uint_t d3) {

#ifdef CUDA_EXAMPLE
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif
        //                   strides  1 x xy
        //                      dims  x y z
        typedef gridtools::layout_map< 0, 1, 2 > layout_t;
        typedef BACKEND::storage_info< 0, layout_t > metadata_t;
        typedef BACKEND::storage_type< float_type, metadata_t >::type storage_type;
        typedef storage_type::pointer_type pointer_type;

        typedef gridtools::halo_exchange_dynamic_ut< layout_t,
            gridtools::layout_map< 0, 1, 2 >,
            pointer_type::pointee_t,
            MPI_3D_process_grid_t< 3 >,
#ifdef CUDA_EXAMPLE
            gridtools::gcl_gpu,
#else
            gridtools::gcl_cpu,
#endif
            gridtools::version_manual > pattern_type;

        pattern_type he(pattern_type::grid_type::period_type(true, false, false), GCL_WORLD);
#ifdef VERBOSE
        printf("halo exchange ok\n");
#endif

        /* The nice interface does not compile today (CUDA 6.5) with nvcc (C++11 support not complete yet)*/

        // Definition of placeholders. The order of them reflect the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain
        typedef arg< 0, storage_type > p_in;
        typedef arg< 1, storage_type > p_out;
        // An array of placeholders to be passed to the domain
        // I'm using mpl::vector, but the final API should look slightly simpler
        typedef boost::mpl::vector< p_in, p_out > accessor_list;
        /* typedef arg<1, vec_storage_type > p_out; */
        // Definition of the actual data fields that are used for input/output
        //#ifdef CXX11_ENABLED
        array< ushort_t, 3 > padding{0, 0, 0};
        array< ushort_t, 3 > halo{1, 1, 1};
        typedef partitioner_trivial< cell_topology< topology::cartesian< layout_map< 0, 1, 2 > > >,
            pattern_type::grid_type > partitioner_t;
        partitioner_t part(he.comm(), halo, padding);
        parallel_storage_info< metadata_t, partitioner_t > meta_(part, d1, d2, d3);
        auto &metadata_ = meta_.get_metadata();

        storage_type in(metadata_, 0.);
        storage_type out(metadata_, 0.);

        he.add_halo< 0 >(meta_.template get_halo_gcl< 0 >());
        he.add_halo< 1 >(meta_.template get_halo_gcl< 1 >());
        // he.add_halo<0>(1,1,1,d1,d1+2);
        // he.add_halo<1>(1,1,1,d2,d2+2);

        he.add_halo< 2 >(0, 0, 0, d3 - 1, d3);

        he.setup(2);

#ifdef VERBOSE
        printf("halo set up\n");
#endif

        for (uint_t i = 0; i < metadata_.template dim< 0 >(); ++i)
            for (uint_t j = 0; j < metadata_.template dim< 1 >(); ++j)
                for (uint_t k = 0; k < metadata_.template dim< 2 >(); ++k) {
                    in(i, j, k) = (i + j + k) * (gridtools::PID + 1);
                }

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        gridtools::grid< axis, partitioner_t > grid(part, meta_);
        // k dimension not partitioned
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
        // that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in
        // order. (I don't particularly like this)
        gridtools::aggregator_type< accessor_list > domain(boost::fusion::make_vector(&in, &out));

/*
  Here we do lot of stuff
  1) We pass to the intermediate representation ::run function the description
  of the stencil, which is a multi-stage stencil (mss)
  The mss includes (in order of execution) a laplacian, two fluxes which are independent
  and a final step that is the out_function
  2) The logical physical domain with the fields to use
  3) The actual domain dimensions
*/

// \todo simplify the following using the auto keyword from C++11
#ifdef CXX11_ENABLED
        auto
#else
#ifdef __CUDACC__
        gridtools::computation *
#else
        boost::shared_ptr< gridtools::computation >
#endif
#endif
            copy = gridtools::make_computation< gridtools::BACKEND >(
                domain,
                grid,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< copy_functor >(p_in() // esf_descriptor
                        ,
                        p_out())));
#ifdef VERBOSE
        printf("computation instantiated\n");
#endif

        copy->ready();

#ifdef VERBOSE
        printf("computation ready\n");
#endif

        copy->steady();

#ifdef VERBOSE
        printf("computation steady\n");
#endif

        copy->run();

#ifdef VERBOSE
        printf("computation run\n");
#endif

        copy->finalize();

#ifdef VERBOSE
        printf("computation finalized\n");
#endif

        std::vector< pointer_type::pointee_t * > vec(2);
        vec[0] = in.data().get();
        vec[1] = out.data().get();

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
        GCL_Finalize();

        printf("copy parallel test executed\n");

        return true;
    }

} // namespace copy_stencil
