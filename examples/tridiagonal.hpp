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

#include <gridtools.hpp>

#include <stencil-composition/stencil-composition.hpp>

#include <stencil-composition/interval.hpp>
#include <stencil-composition/make_computation.hpp>
#include <tools/verifier.hpp>

#include "backend_select.hpp"

/*
  @file This file shows an implementation of the Thomas algorithm, done using stencil operations.

  Important convention: the linear system as usual is represented with 4 vectors: the main diagonal
  (diag), the upper and lower first diagonals (sup and inf respectively), and the right hand side
  (rhs). Note that the dimensions and the memory layout are, for an NxN system
  rank(diag)=N       [xxxxxxxxxxxxxxxxxxxxxxxx]
  rank(inf)=N-1      [0xxxxxxxxxxxxxxxxxxxxxxx]
  rank(sup)=N-1      [xxxxxxxxxxxxxxxxxxxxxxx0]
  rank(rhs)=N        [xxxxxxxxxxxxxxxxxxxxxxxx]
  where x denotes any number and 0 denotes the padding, a dummy value which is not used in
  the algorithm. This choice coresponds to having the same vector index for each row of the matrix.
 */

using gridtools::level;
using gridtools::accessor;
using gridtools::extent;
using gridtools::arg;

namespace tridiagonal {

    using namespace gridtools;
    using namespace enumtype;

    using namespace expressions;

    // This is the definition of the special regions in the "vertical" direction
    using axis_t = axis< 1 >;
    using x_internal = axis_t::full_interval::modify< 1, -1 >;
    using x_first = axis_t::full_interval::first_level;
    using x_last = axis_t::full_interval::last_level;

    typedef dimension< 3 > z;

    struct forward_thomas {
        // four vectors: output, and the 3 diagonals
        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1 > inf;                  // a
        typedef accessor< 2 > diag;                 // b
        typedef accessor< 3, enumtype::inout > sup; // c
        typedef accessor< 4, enumtype::inout > rhs; // d
        typedef boost::mpl::vector< out, inf, diag, sup, rhs > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void shared_kernel(Evaluation &eval) {
            eval(sup{}) = eval(sup{} / (diag{} - sup{z{-1}} * inf{}));
            eval(rhs{}) = eval((rhs{} - inf{} * rhs{z(-1)}) / (diag{} - sup{z(-1)} * inf{}));
        }

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_internal) {
            shared_kernel(eval);
        }

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_last) {
            shared_kernel(eval);
        }

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_first) {
            eval(sup()) = eval(sup()) / eval(diag());
            eval(rhs()) = eval(rhs()) / eval(diag());
        }
    };

    struct backward_thomas {
        typedef accessor< 0, enumtype::inout > out;
        typedef accessor< 1 > inf;                  // a
        typedef accessor< 2 > diag;                 // b
        typedef accessor< 3, enumtype::inout > sup; // c
        typedef accessor< 4, enumtype::inout > rhs; // d
        typedef boost::mpl::vector< out, inf, diag, sup, rhs > arg_list;

        template < typename Evaluation >
        GT_FUNCTION static void shared_kernel(Evaluation &eval) {
            eval(out()) = eval(rhs{} - sup{} * out{0, 0, 1});
        }

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_internal) {
            shared_kernel(eval);
        }

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_first) {
            shared_kernel(eval);
        }

        template < typename Evaluation >
        GT_FUNCTION static void Do(Evaluation &eval, x_last) {
            eval(out()) = eval(rhs());
        }
    };

    std::ostream &operator<<(std::ostream &s, backward_thomas const) { return s << "backward_thomas"; }
    std::ostream &operator<<(std::ostream &s, forward_thomas const) { return s << "forward_thomas"; }

    bool test(uint_t d1, uint_t d2, uint_t d3) {

        if (d3 != 6)
            std::cout << "WARNING: This test is only working with 6 k levels,"
                         "to guarantee that result can be validated to 1"
                      << std::endl;
        d3 = 6;

        typedef backend_t::storage_traits_t::storage_info_t< 0, 3 > meta_t;
        typedef backend_t::storage_traits_t::data_store_t< float_type, meta_t > storage_type;

        // Definition of the actual data fields that are used for input/output
        meta_t meta_(d1, d2, d3);
        storage_type out(meta_, 0.0, "out");
        storage_type inf(meta_, -1.0, "inf");
        storage_type diag(meta_, 3.0, "diag");
        storage_type sup(meta_, 1.0, "sup");
        storage_type rhs(meta_, 3.0, "rhs");
        storage_type solution(meta_, 1.0, "solution");

        // special field initalizations
        auto rhsv = make_host_view(rhs);
        for (int_t i = 0; i < d1; ++i) {
            for (int_t j = 0; j < d2; ++j) {
                rhsv(i, j, 0) = 4.;
                rhsv(i, j, 5) = 2.;
            }
        }
        // result is 1

        // Definition of placeholders. The order of them reflect the order the user will deal with them
        // especially the non-temporary ones, in the construction of the domain
        typedef arg< 0, storage_type > p_inf;  // a
        typedef arg< 1, storage_type > p_diag; // b
        typedef arg< 2, storage_type > p_sup;  // c
        typedef arg< 3, storage_type > p_rhs;  // d
        typedef arg< 4, storage_type > p_out;

        // An array of placeholders to be passed to the domain
        // I'm using mpl::vector, but the final API should look slightly simpler
        typedef boost::mpl::vector< p_inf, p_diag, p_sup, p_rhs, p_out > accessor_list;

        // construction of the domain. The domain is the physical domain of the problem, with all the physical fields
        // that are used, temporary and not
        // It must be noted that the only fields to be passed to the constructor are the non-temporary.
        // The order in which they have to be passed is the order in which they appear scanning the placeholders in
        // order. (I don't particularly like this)
        gridtools::aggregator_type< accessor_list > domain(inf, diag, sup, rhs, out);

        auto grid = make_grid(d1, d2, axis_t(d3));

        /*
          Here we do lot of stuff
          1) We pass to the intermediate representation ::run function the description
          of the stencil, which is a multi-stage stencil (mss)
          The mss includes (in order of execution) a laplacian, two fluxes which are independent
          and a final step that is the out_function
          2) The logical physical domain with the fields to use
          3) The actual domain dimensions
         */

        auto solver = gridtools::make_computation< backend_t >(
            domain,
            grid,
            gridtools::make_multistage // mss_descriptor
            (execute< forward >(),
                gridtools::make_stage< forward_thomas >(p_out(), p_inf(), p_diag(), p_sup(), p_rhs()) // esf_descriptor
                ),
            gridtools::make_multistage // mss_descriptor
            (execute< backward >(),
                gridtools::make_stage< backward_thomas >(p_out(), p_inf(), p_diag(), p_sup(), p_rhs()) // esf_descriptor
                ));

        solver.steady();

        solver.run();

        solver.finalize();

#ifdef BENCHMARK
        std::cout << solver.print_meter() << std::endl;
#endif

#if FLOAT_PRECISION == 4
        verifier verif(1e-6);
#else
        verifier verif(1e-12);
#endif
        array< array< uint_t, 2 >, 3 > halos{{{0, 0}, {0, 0}, {0, 0}}};
        bool result = verif.verify(grid, solution, out, halos);

        return result;
    }
} // namespace tridiagonal
