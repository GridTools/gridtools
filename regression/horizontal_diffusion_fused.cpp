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

#include <gridtools/stencil_composition/cartesian.hpp>
#include <gridtools/tools/cartesian_regression_fixture.hpp>

#include "horizontal_diffusion_repository.hpp"

using namespace gridtools;
using namespace cartesian;

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
    auto out = make_storage();

    horizontal_diffusion_repository repo(d(0), d(1), d(2));

    auto comp = [grid = make_grid(), &out, in = make_storage(repo.in), coeff = make_storage(repo.coeff)] {
        run_single_stage(out_function(), backend_t(), grid, out, in, coeff);
    };

    comp();
    verify(repo.out, out);
    benchmark(comp);
}
