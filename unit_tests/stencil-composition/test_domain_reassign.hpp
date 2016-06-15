/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once
#include <stencil_composition/stencil_composition.hpp>
#include "../../examples/cache_flusher.hpp"
#include "../../examples/defs.hpp"

/**
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

namespace domain_reassign {
#ifdef __CUDACC__
    typedef gridtools::layout_map< 2, 1, 0 > layout_t; // stride 1 on i
#else
    //                   strides  1 x xy
    //                      dims  x y z
    typedef gridtools::layout_map< 0, 1, 2 > layout_t; // stride 1 on k
#endif

    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    struct test_functor {

        typedef accessor< 0, enumtype::in, extent<>, 3 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
        typedef boost::mpl::vector< in, out > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation const &eval, x_interval) {
            eval(out()) = eval(in());
        }
    };

    /*
     * The following operators and structs are for debugging only
     */
    std::ostream &operator<<(std::ostream &s, test_functor const) { return s << "test_functor"; }

    void handle_error(int_t) { std::cout << "error" << std::endl; }

    bool test() {

        cache_flusher flusher(cache_flusher_size);

        uint_t d1 = 32;
        uint_t d2 = 32;
        uint_t d3 = 32;

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

        meta_data_t meta_data_(d1, d2, d3);

        // Definition of the actual data fields that are used for input/output
        typedef storage_t storage_type;
        storage_type in(meta_data_, "in");
        storage_type out(meta_data_, float_type(-1.));
        storage_type in2(meta_data_, "in2");
        storage_type out2(meta_data_, float_type(-1.));
        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k) {
                    in(i, j, k) = i + j + k;
                    in2(i, j, k) = i + j + k + 3;
                }

        typedef arg< 0, storage_type > p_in;
        typedef arg< 1, storage_type > p_out;

        typedef boost::mpl::vector< p_in, p_out > accessor_list;
        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
        // that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in
        // order. (I don't particularly like this)
        gridtools::aggregator_type< accessor_list > domain((p_in() = in), (p_out() = out));

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        /*
          Here we do lot of stuff
          1) We pass to the intermediate representation ::run function the description
          of the stencil, which is a multi-stage stencil (mss)
          The mss includes (in order of execution) a laplacian, two fluxes which are independent
          and a final step that is the out_function
          2) The logical physical domain with the fields to use
          3) The actual domain dimensions
        */

        auto copy = gridtools::make_computation< gridtools::BACKEND >(
            domain,
            grid,
            gridtools::make_multistage // mss_descriptor
            (execute< forward >(),
                gridtools::make_stage< test_functor >(p_in() // esf_descriptor
                    ,
                    p_out())));

        copy->ready();

        copy->steady();

        copy->run();

#ifdef __CUDACC__
        out.d2h_update();
        in.d2h_update();
#endif
        copy->finalize();

        bool success = true;
        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k) {
                    if (in(i, j, k) != out(i, j, k)) {
                        std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                  << "in = " << in(i, j, k) << ", out = " << out(i, j, k) << std::endl;
                        success = false;
                    }
                }

        copy->reassign((p_in() = in2), (p_out() = out2));

        copy->ready();

        copy->steady();

        copy->run();

        copy->finalize();

#ifdef __CUDACC__
        out.d2h_update();
        in.d2h_update();
#endif

        for (uint_t i = 0; i < d1; ++i)
            for (uint_t j = 0; j < d2; ++j)
                for (uint_t k = 0; k < d3; ++k) {
                    if (in2(i, j, k) != out2(i, j, k)) {
                        std::cout << "error in " << i << ", " << j << ", " << k << ": "
                                  << "in = " << in2(i, j, k) << ", out = " << out2(i, j, k) << std::endl;
                        success = false;
                    }
                }

        return success;
    }
} // namespace
