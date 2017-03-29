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
#include <tools/verifier.hpp>

/*
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

using namespace gridtools;
using namespace enumtype;

static const int _value_ = 1;

namespace positional_copy_stencil {
#ifdef __CUDACC__
    typedef gridtools::layout_map< 2, 1, 0 > layout_t; // stride 1 on i
#else
    typedef gridtools::layout_map< 0, 1, 2 > layout_t; // stride 1 on k
#endif

    // This is the definition of the special regions in the "vertical" direction
    typedef gridtools::interval< level< 0, -1 >, level< 1, -1 > > x_interval;
    typedef gridtools::interval< level< 0, -2 >, level< 1, 1 > > axis;

    // These are the stencil operators that compose the multistage stencil in this test
    template < int V >
    struct init_functor {
        typedef accessor< 0, enumtype::inout, extent<> > one;
        typedef accessor< 1, enumtype::inout, extent<> > two;
        typedef boost::mpl::vector< one, two > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_interval) {
            eval(one()) = static_cast< float_type >(V) * (eval.i() + eval.j() + eval.k());
            eval(two()) = -1.1;
        }
    };

    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor {

        typedef accessor< 0, enumtype::in, extent<>, 3 > in;
        typedef accessor< 1, enumtype::inout, extent<>, 3 > out;
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

    /*
     * The following operators and structs are for debugging only
     */
    template < int I >
    std::ostream &operator<<(std::ostream &s, init_functor< I > const) {
        return s << "(positional) init_functor";
    }

    void handle_error(int_t) { std::cout << "error" << std::endl; }

    bool test(uint_t x, uint_t y, uint_t z) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

#ifdef __CUDACC__
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
        typedef gridtools::layout_map< 2, 1, 0 > layout_t;
        typedef gridtools::BACKEND::storage_info< 0, layout_t > meta_t;

        typedef gridtools::BACKEND::storage_type< float_type, meta_t >::type storage_type;

        // Definition of placeholders. The order of them reflect the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain

        typedef arg< 0, storage_type > p_in;
        typedef arg< 1, storage_type > p_out;
        // An array of placeholders to be passed to the domain
        // I'm using mpl::vector, but the final API should look slightly simpler
        typedef boost::mpl::vector< p_in, p_out > accessor_list;

        /* typedef arg<1, vec_field_type > p_out; */

        // Definition of the actual data fields that are used for input/output
        meta_t meta_(d1, d2, d3);
        storage_type in(meta_, -3.5, "in");
        storage_type out(meta_, 1.5, "out");

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
        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

#ifdef CXX11_ENABLED
        auto
#else
#ifdef __CUDACC__
        gridtools::stencil *
#else
        boost::shared_ptr< gridtools::stencil >
#endif
#endif
            init = gridtools::make_positional_computation< gridtools::BACKEND >(
                domain,
                grid,
                gridtools::make_multistage // mss_descriptor
                (execute< forward >(),
                    gridtools::make_stage< init_functor< _value_ > >(p_in(), p_out() // esf_descriptor
                        )));

        init->ready();

        init->steady();
        domain.clone_to_device();
        init->run();

        init->finalize();

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
                    gridtools::make_stage< copy_functor >(p_in() // esf_descriptor
                        ,
                        p_out())));

        copy->ready();

        copy->steady();

        copy->run();

        copy->finalize();

        storage_type ref(meta_, 1.5, "ref");

        for (uint_t i = 0; i < d1; ++i) {
            for (uint_t j = 0; j < d2; ++j) {
                for (uint_t k = 0; k < d3; ++k) {
                    ref(i, j, k) = static_cast< double >(_value_) * (i + j + k);
                }
            }
        }

#ifdef CXX11_ENABLED
#if FLOAT_PRECISION == 4
        verifier verif(1e-6);
#else
        verifier verif(1e-12);
#endif
        array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};
        bool result = verif.verify(grid, ref, out, halos);
#else
#if FLOAT_PRECISION == 4
        verifier verif(1e-6, 0);
#else
        verifier verif(1e-12, 0);
#endif
        bool result = verif.verify(grid, ref, out);
#endif
        return result;
    }

} // namespace copy_stencil
