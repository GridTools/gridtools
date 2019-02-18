/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
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
using full_t = axis_t::full_interval;

struct forward_thomas {
    // four vectors: output, and the 3 diagonals
    using out = inout_accessor<0>;
    using inf = in_accessor<1>;    // a
    using diag = in_accessor<2>;   // b
    using sup = inout_accessor<3>; // c
    using rhs = inout_accessor<4>; // d
    using param_list = make_param_list<out, inf, diag, sup, rhs>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, 0>) {
        eval(sup{}) = eval(sup{} / (diag{} - sup{0, 0, -1} * inf{}));
        eval(rhs{}) = eval((rhs{} - inf{} * rhs{0, 0, -1}) / (diag{} - sup{0, 0, -1} * inf{}));
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
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
    using param_list = make_param_list<out, inf, diag, sup, rhs>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -1>) {
        eval(out{}) = eval(rhs{} - sup{} * out{0, 0, 1});
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        eval(out{}) = eval(rhs{});
    }
};

using tridiagonal = regression_fixture<>;

TEST_F(tridiagonal, test) {
    d3() = 6;

    auto out = make_storage();
    auto sup = make_storage(1.);
    auto rhs = make_storage([](int_t, int_t, int_t k) { return k == 0 ? 4. : k == 5 ? 2. : 3.; });

    arg<0> p_inf;  // a
    arg<1> p_diag; // b
    arg<2> p_sup;  // c
    arg<3> p_rhs;  // d
    arg<4> p_out;

    make_computation(p_inf = make_storage(-1.),
        p_diag = make_storage(3.),
        p_sup = sup,
        p_rhs = rhs,
        p_out = out,
        make_multistage(execute::forward(), make_stage<forward_thomas>(p_out, p_inf, p_diag, p_sup, p_rhs)),
        make_multistage(execute::backward(), make_stage<backward_thomas>(p_out, p_inf, p_diag, p_sup, p_rhs)))
        .run();

    verify(make_storage(1.), out);
}
