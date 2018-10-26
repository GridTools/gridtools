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
#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

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

using namespace gridtools;
using namespace expressions;

// This is the definition of the special regions in the "vertical" direction
using axis_t = axis<1>;
using x_internal = axis_t::full_interval::modify<1, -1>;
using x_first = axis_t::full_interval::first_level;
using x_last = axis_t::full_interval::last_level;

struct forward_thomas {
    // four vectors: output, and the 3 diagonals
    using out = inout_accessor<0>;
    using inf = in_accessor<1>;    // a
    using diag = in_accessor<2>;   // b
    using sup = inout_accessor<3>; // c
    using rhs = inout_accessor<4>; // d
    using arg_list = boost::mpl::vector<out, inf, diag, sup, rhs>;

    template <typename Evaluation>
    GT_FUNCTION static void shared_kernel(Evaluation &eval) {
        eval(sup{}) = eval(sup{} / (diag{} - sup{0, 0, -1} * inf{}));
        eval(rhs{}) = eval((rhs{} - inf{} * rhs{0, 0, -1}) / (diag{} - sup{0, 0, -1} * inf{}));
    }

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval, x_internal) {
        shared_kernel(eval);
    }

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval, x_last) {
        shared_kernel(eval);
    }

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval, x_first) {
        eval(sup{}) = eval(sup{}) / eval(diag{});
        eval(rhs{}) = eval(rhs{}) / eval(diag{});
    }
};

struct backward_thomas {
    using out = inout_accessor<0>;
    using inf = in_accessor<1>;    // a
    using diag = in_accessor<2>;   // b
    using sup = inout_accessor<3>; // c
    using rhs = inout_accessor<4>; // d
    using arg_list = boost::mpl::vector<out, inf, diag, sup, rhs>;

    template <typename Evaluation>
    GT_FUNCTION static void shared_kernel(Evaluation &eval) {
        eval(out{}) = eval(rhs{} - sup{} * out{0, 0, 1});
    }

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval, x_internal) {
        shared_kernel(eval);
    }

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval, x_first) {
        shared_kernel(eval);
    }

    template <typename Evaluation>
    GT_FUNCTION static void Do(Evaluation eval, x_last) {
        eval(out{}) = eval(rhs{});
    }
};

using TridiagonalSolve = regression_fixture<>;

TEST_F(TridiagonalSolve, Test) {
    d3() = 6;

    auto out = make_storage(0.);
    auto sup = make_storage(1.);
    auto rhs = make_storage([](int_t, int_t, int_t k) { return k == 0 ? 4. : k == 5 ? 2. : 3.; });

    arg<0, storage_type> p_inf;  // a
    arg<1, storage_type> p_diag; // b
    arg<2, storage_type> p_sup;  // c
    arg<3, storage_type> p_rhs;  // d
    arg<4, storage_type> p_out;

    make_computation(p_inf = make_storage(-1.),
        p_diag = make_storage(3.),
        p_sup = sup,
        p_rhs = rhs,
        p_out = out,
        make_multistage(
            enumtype::execute<enumtype::forward>(), make_stage<forward_thomas>(p_out, p_inf, p_diag, p_sup, p_rhs)),
        make_multistage(
            enumtype::execute<enumtype::backward>(), make_stage<backward_thomas>(p_out, p_inf, p_diag, p_sup, p_rhs)))
        .run();

    verify(make_storage(1.), out);
}
