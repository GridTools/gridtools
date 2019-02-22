/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// TODO(fthaler): merge this with horizontal_diffusion_functions and benchmark all the horizontal_diffusion versions

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/stencil-composition/stencil-functions/stencil-functions.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "horizontal_diffusion_repository.hpp"

using namespace gridtools;

struct lap_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 1>>;

    using param_list = make_param_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) =
            float_type{4} * eval(in()) - (eval(in(1, 0)) + eval(in(0, 1)) + eval(in(-1, 0)) + eval(in(0, -1)));
    }
};

struct flx_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 2, -1, 1>>;

    using param_list = make_param_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        auto lap_hi = call<lap_function>::with(eval, in(1, 0));
        auto lap_lo = call<lap_function>::with(eval, in(0, 0));
        auto flx = lap_hi - lap_lo;
        eval(out()) = flx * (eval(in(1, 0)) - eval(in(0, 0))) > 0 ? 0 : flx;
    }
};

struct fly_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 2>>;

    using param_list = make_param_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        auto lap_hi = call<lap_function>::with(eval, in(0, 1));
        auto lap_lo = call<lap_function>::with(eval, in(0, 0));
        auto fly = lap_hi - lap_lo;
        eval(out()) = fly * (eval(in(0, 1)) - eval(in(0, 0))) > 0 ? 0 : fly;
    }
};

struct out_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-2, 2, -2, 2>>;
    using coeff = in_accessor<2>;

    using param_list = make_param_list<out, in, coeff>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        auto flx_hi = call<flx_function>::with(eval, in(0, 0));
        auto flx_lo = call<flx_function>::with(eval, in(-1, 0));

        auto fly_hi = call<fly_function>::with(eval, in(0, 0));
        auto fly_lo = call<fly_function>::with(eval, in(0, -1));

        eval(out()) = eval(in()) - eval(coeff()) * (flx_hi - flx_lo + fly_hi - fly_lo);
    }
};

using horizontal_diffusion_fused = regression_fixture<2>;

TEST_F(horizontal_diffusion_fused, test) {
    arg<0> p_coeff;
    arg<1> p_in;
    arg<2> p_out;

    auto out = make_storage();

    horizontal_diffusion_repository repo(d1(), d2(), d3());

    auto comp = make_computation(p_in = make_storage(repo.in),
        p_out = out,
        p_coeff = make_storage(repo.coeff),
        make_multistage(execute::parallel(), make_stage<out_function>(p_out, p_in, p_coeff)));

    comp.run();
    verify(make_storage(repo.out), out);
    benchmark(comp);
}
