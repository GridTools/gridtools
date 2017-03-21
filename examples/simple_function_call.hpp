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
#include <stencil-composition/structured_grids/call_interfaces.hpp>
#include "defs.hpp"
#include "benchmarker.hpp"

using namespace gridtools;
using namespace enumtype;

namespace simple_function_call {
#ifdef __CUDACC__
    typedef gridtools::layout_map< 2, 1, 0 > layout_t; // stride 1 on i
#else
    typedef gridtools::layout_map< 0, 1, 2 > layout_t; // stride 1 on k
#endif

    typedef gridtools::interval< level< 0, -1 >, level< 1, -2 > > lower;
    typedef gridtools::interval< level< 1, -1 >, level< 2, -1 > > upper;
    typedef gridtools::interval< level< 0, -1 >, level< 2, -1 > > full_domain;

    typedef gridtools::interval< level< 0, -1 >, level< 2, 1 > > axis;

    struct delta {
        typedef accessor< 0, enumtype::in, extent< 0, 1, 0, 0, 0, 0 >, 3 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval) {
            eval(out()) = eval(in(1, 0, 0)) - eval(in());
        }
    };

    struct multiply_depending_on_interval {
        typedef accessor< 0, enumtype::in, extent<>, 3 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        // call average on the lower
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, lower) {
            eval(out()) = call< delta >::with(eval, in());
        }
        // call average on the upper do something additional
        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, upper) {
            eval(out()) = 2. * call< delta >::with_offsets(eval, in(0, 1, 0));
        }
    };

    void handle_error(int_t) { std::cout << "error" << std::endl; }

    bool test(uint_t x, uint_t y, uint_t z, uint_t t_steps, bool verify) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        uint_t kmax = z / 2;

#ifdef __CUDACC__
#define BACKEND backend< Cuda, GRIDBACKEND, Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< Host, GRIDBACKEND, Block >
#else
#define BACKEND backend< Host, GRIDBACKEND, Naive >
#endif
#endif

        typedef BACKEND::storage_info< __COUNTER__, layout_t > meta_data_t;
        typedef BACKEND::storage_type< float_type, meta_data_t >::type storage_t;

        meta_data_t meta_data_(x, y, z);

        // Definition of the actual data fields that are used for input/output
        typedef storage_t storage_type;
        storage_type in(meta_data_, "in");
        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k) {
                    in(i, j, k) = i + j + k;
                }
        storage_type out(meta_data_, float_type(-1.));

        typedef arg< 0, storage_type > p_in;
        typedef arg< 1, storage_type > p_out;

        typedef boost::mpl::vector< p_in, p_out > accessor_list;
        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
        // that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in
        // order. (I don't particularly like this)
        gridtools::aggregator_type< accessor_list > domain(boost::fusion::make_vector(&in, &out));

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
        uint_t halo_i = 1;
        uint_t di[5] = {0, halo_i, 0, d1 - halo_i - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = kmax;
        grid.value_list[2] = d3 - 1;

/*
  Here we do lot of stuff
  1) We pass to the intermediate representation ::run function the description
  of the stencil, which is a multi-stage stencil (mss)
  The mss includes (in order of execution) a laplacian, two fluxes which are independent
  and a final step that is the out_function
  2) The logical physical domain with the fields to use
  3) The actual domain dimensions
*/

#ifdef CXX11_ENABLED
        auto
#else
#ifdef __CUDACC__
        gridtools::stencil *
#else
        boost::shared_ptr< gridtools::stencil >
#endif
#endif
            copy = gridtools::make_computation< gridtools::BACKEND >(
                domain,
                grid,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< multiply_depending_on_interval >(p_in() // esf_descriptor
                        ,
                        p_out())));

        copy->ready();

        copy->steady();

        copy->run();

#ifdef __CUDACC__
        out.d2h_update();
        in.d2h_update();
#endif

        storage_t reference(meta_data_, 1.);
        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k) {
                    if (k < kmax)
                        reference(i, j, k) = 1.;
                    else
                        reference(i, j, k) = 2.;
                }

        bool success = true;
        if (verify) {
            for (uint_t i = 0; i < d1 - halo_i; ++i) {
                for (uint_t j = 0; j < d2; ++j) {
                    for (uint_t k = 0; k < d3; ++k) {
                        if (reference(i, j, k) != out(i, j, k)) {
                            std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                      << "in = " << in(i, j, k) << ", reference = " << reference(i, j, k) << std::endl;
                            success = false;
                        }
                    }
                }
            }
        }
#ifdef BENCHMARK
        benchmarker::run(copy, t_steps);
#endif
        copy->finalize();

        return success;
    }
} // namespace copy_stencil
