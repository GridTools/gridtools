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

#include "backend_select.hpp"
#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/verifier.hpp>

/*
  @file
  This file shows an implementation of the "copy" stencil, simple copy of one field done on the backend
*/

using gridtools::accessor;
using gridtools::arg;
using gridtools::extent;
using gridtools::level;

using namespace gridtools;
using namespace enumtype;

static const int _value_ = 1;

namespace positional_copy_stencil {
    // These are the stencil operators that compose the multistage stencil in this test
    template <int V>
    struct init_functor {
        typedef accessor<0, enumtype::inout, extent<>> one;
        typedef accessor<1, enumtype::inout, extent<>> two;
        typedef boost::mpl::vector<one, two> arg_list;

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation &eval) {
            eval(one()) = static_cast<float_type>(V) * (eval.i() + eval.j() + eval.k());
            eval(two()) = -1.1;
        }
    };

    // These are the stencil operators that compose the multistage stencil in this test
    struct copy_functor {

        typedef accessor<0, enumtype::in, extent<>, 3> in;
        typedef accessor<1, enumtype::inout, extent<>, 3> out;
        typedef boost::mpl::vector<in, out> arg_list;

        /* static const auto expression=in(1,0,0)-out(); */

        template <typename Evaluation>
        GT_FUNCTION static void Do(Evaluation &eval) {
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
    template <int I>
    std::ostream &operator<<(std::ostream &s, init_functor<I> const) {
        return s << "(positional) init_functor";
    }

    void handle_error(int_t) { std::cout << "error" << std::endl; }

    bool test(uint_t x, uint_t y, uint_t z) {

        uint_t d1 = x;
        uint_t d2 = y;
        uint_t d3 = z;

        typedef backend_t::storage_traits_t::storage_info_t<0, 3> meta_data_t;
        typedef backend_t::storage_traits_t::data_store_t<float_type, meta_data_t> storage_t;

        // Definition of placeholders. The order of them reflect the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain

        typedef arg<0, storage_t> p_in;
        typedef arg<1, storage_t> p_out;

        /* typedef arg<1, vec_field_type > p_out; */

        // Definition of the actual data fields that are used for input/output
        meta_data_t meta_(d1, d2, d3);
        storage_t in(meta_, -3.5, "in");
        storage_t out(meta_, 1.5, "out");

        auto grid = make_grid(d1, d2, d3);

        auto init = gridtools::make_positional_computation<backend_t>(grid,
            p_in{} = in,
            p_out{} = out,
            gridtools::make_multistage // mss_descriptor
            (execute<forward>(),
                gridtools::make_stage<init_functor<_value_>>(p_in(), p_out() // esf_descriptor
                    )));

        init.run();
        init.sync_bound_data_stores();

        auto copy = gridtools::make_computation<backend_t>(grid,
            p_in{} = in,
            p_out{} = out,
            gridtools::make_multistage // mss_descriptor
            (execute<forward>(),
                gridtools::make_stage<copy_functor>(p_in() // esf_descriptor
                    ,
                    p_out())));
        copy.run();
        copy.sync_bound_data_stores();

        storage_t ref(meta_, [](int i, int j, int k) { return static_cast<double>(_value_) * (i + j + k); });

#if FLOAT_PRECISION == 4
        verifier verif(1e-6);
#else
        verifier verif(1e-12);
#endif
        array<array<uint_t, 2>, 3> halos{{{0, 0}, {0, 0}, {0, 0}}};
        bool result = verif.verify(grid, ref, out, halos);
        return result;
    }

} // namespace positional_copy_stencil
