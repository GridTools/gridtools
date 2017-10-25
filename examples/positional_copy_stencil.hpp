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

#ifdef CUDA_EXAMPLE
#define BACKEND backend< enumtype::Cuda, enumtype::GRIDBACKEND, enumtype::Block >
#elif defined(__AVX512F__)
#define BACKEND backend< enumtype::Mic, enumtype::GRIDBACKEND, enumtype::Block >
#else
#ifdef BACKEND_BLOCK
#define BACKEND backend< enumtype::Host, enumtype::GRIDBACKEND, enumtype::Block >
#else
#define BACKEND backend< enumtype::Host, enumtype::GRIDBACKEND, enumtype::Naive >
#endif
#endif

namespace positional_copy_stencil {

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

        typedef BACKEND::storage_traits_t::storage_info_t< 0, 3 > meta_data_t;
        typedef BACKEND::storage_traits_t::data_store_t< float_type, meta_data_t > storage_t;

        // Definition of placeholders. The order of them reflect the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain

        typedef arg< 0, storage_t > p_in;
        typedef arg< 1, storage_t > p_out;
        // An array of placeholders to be passed to the domain
        // I'm using mpl::vector, but the final API should look slightly simpler
        typedef boost::mpl::vector< p_in, p_out > accessor_list;

        /* typedef arg<1, vec_field_type > p_out; */

        // Definition of the actual data fields that are used for input/output
        meta_data_t meta_(d1, d2, d3);
        storage_t in(meta_, -3.5, "in");
        storage_t out(meta_, 1.5, "out");

        gridtools::aggregator_type< accessor_list > domain(in, out);

        // Definition of the physical dimensions of the problem.
        // The constructor takes the horizontal plane dimensions,
        // while the vertical ones are set according the the axis property soon after
        // gridtools::grid<axis> grid(2,d1-2,2,d2-2);
        uint_t di[5] = {0, 0, 0, d1 - 1, d1};
        uint_t dj[5] = {0, 0, 0, d2 - 1, d2};

        gridtools::grid< axis > grid(di, dj);
        grid.value_list[0] = 0;
        grid.value_list[1] = d3 - 1;

        auto init = gridtools::make_positional_computation< gridtools::BACKEND >(
            domain,
            grid,
            gridtools::make_multistage // mss_descriptor
            (execute< forward >(),
                gridtools::make_stage< init_functor< _value_ > >(p_in(), p_out() // esf_descriptor
                    )));

        init->ready();
        init->steady();
        init->run();
        init->finalize();

        auto copy = gridtools::make_computation< gridtools::BACKEND >(
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

        storage_t ref(meta_, [](int i, int j, int k) { return static_cast< double >(_value_) * (i + j + k); });

#if FLOAT_PRECISION == 4
        verifier verif(1e-6);
#else
        verifier verif(1e-12);
#endif
        array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};
        bool result = verif.verify(grid, ref, out, halos);
        return result;
    }

} // namespace copy_stencil
