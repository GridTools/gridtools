/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "interpolate_stencil.hpp"

#include <gridtools/stencil_composition/expressions/expressions.hpp>
#include <gridtools/stencil_composition/global_parameter.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>

namespace {
    using namespace gridtools;

    struct interpolate_stage {
        using in1 = in_accessor<0>;
        using in2 = in_accessor<1>;
        using weight = in_accessor<2>;
        using out = inout_accessor<3>;

        using param_list = make_param_list<in1, in2, weight, out>;

        template <class Eval>
        GT_FUNCTION static void apply(Eval &&eval) {
            using namespace expressions;
            eval(out()) = eval(weight() * in1() + (1. - weight()) * in2());
        }
    };
} // namespace

// `make_computation` should never be called in a header, because the compilation overhead is very significant
std::function<void(inputs const &, outputs const &)> make_interpolate_stencil(grid_t grid, double weight) {
    return [grid = std::move(grid), weight](inputs const &in, outputs &out) {
        arg<0> p_in1;
        arg<1> p_in2;
        arg<2> p_out;
        arg<3> p_weight;
        compute<backend_t>(grid,
            p_weight = make_global_parameter(weight),
            p_in1 = in.in1,
            p_in2 = in.in2,
            p_out = out.out,
            make_multistage(execute::parallel(), make_stage<interpolate_stage>(p_in1, p_in2, p_weight, p_out)));
    };
}
