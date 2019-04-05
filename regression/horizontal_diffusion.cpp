/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

#include "horizontal_diffusion_repository.hpp"

using namespace gridtools;

struct lap_function {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(out), GT_IN_ACCESSOR(in, extent<-1, 1, -1, 1>));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) =
            float_type{4} * eval(in()) - (eval(in(1, 0)) + eval(in(0, 1)) + eval(in(-1, 0)) + eval(in(0, -1)));
    }
};

struct flx_function {
    GT_DEFINE_ACCESSORS(
        GT_INOUT_ACCESSOR(out), GT_IN_ACCESSOR(in, extent<0, 1, 0, 0>), GT_IN_ACCESSOR(lap, extent<0, 1, 0, 0>));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = eval(lap(1, 0)) - eval(lap(0, 0));
        if (eval(out()) * (eval(in(1, 0, 0)) - eval(in(0, 0))) > 0) {
            eval(out()) = 0.;
        }
    }
};

struct fly_function {
    GT_DEFINE_ACCESSORS(
        GT_INOUT_ACCESSOR(out), GT_IN_ACCESSOR(in, extent<0, 0, 0, 1>), GT_IN_ACCESSOR(lap, extent<0, 0, 0, 1>));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = eval(lap(0, 1)) - eval(lap(0, 0));
        if (eval(out()) * (eval(in(0, 1)) - eval(in(0, 0))) > 0)
            eval(out()) = 0.;
    }
};

struct out_function {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(out),
        GT_IN_ACCESSOR(in),
        GT_IN_ACCESSOR(flx, extent<-1, 0, 0, 0>),
        GT_IN_ACCESSOR(fly, extent<0, 0, -1, 0>),
        GT_IN_ACCESSOR(coeff));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = eval(in()) - eval(coeff()) * (eval(flx()) - eval(flx(-1, 0)) + eval(fly()) - eval(fly(0, -1)));
    }
};

using horizontal_diffusion = regression_fixture<2>;

TEST_F(horizontal_diffusion, test) {
    tmp_arg<0> p_lap;
    tmp_arg<1> p_flx;
    tmp_arg<2> p_fly;
    arg<3> p_coeff;
    arg<4> p_in;
    arg<5> p_out;

    auto out = make_storage();

    horizontal_diffusion_repository repo(d1(), d2(), d3());

    auto comp = make_computation(p_in = make_storage(repo.in),
        p_out = out,
        p_coeff = make_storage(repo.coeff),
        make_multistage(execute::parallel(),
            define_caches(cache<cache_type::ij, cache_io_policy::local>(p_lap, p_flx, p_fly)),
            make_stage<lap_function>(p_lap, p_in),
            make_independent(
                make_stage<flx_function>(p_flx, p_in, p_lap), make_stage<fly_function>(p_fly, p_in, p_lap)),
            make_stage<out_function>(p_out, p_in, p_flx, p_fly, p_coeff)));

    comp.run();
    verify(make_storage(repo.out), out);
    benchmark(comp);
}
