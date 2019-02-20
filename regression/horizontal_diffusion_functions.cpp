/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
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
        eval(out()) = 4. * eval(in()) - (eval(in(-1, 0)) + eval(in(0, -1)) + eval(in(0, 1)) + eval(in(1, 0)));
    }
};

enum class variation { monolithic, call, call_offsets, procedures, procedures_offsets };

template <variation, class Acc>
struct lap;

template <typename Acc>
struct lap<variation::monolithic, Acc> {
    template <typename Evaluation>
    GT_FUNCTION static float_type do00(Evaluation &eval) {
        return 4. * eval(Acc{}) - (eval(Acc{-1, 0}) + eval(Acc{0, -1}) + eval(Acc{0, 1}) + eval(Acc{1, 0}));
    }
    template <typename Evaluation>
    GT_FUNCTION static float_type do10(Evaluation &eval) {
        return 4. * eval(Acc{1, 0}) - (eval(Acc{0, 0}) + eval(Acc{1, -1}) + eval(Acc{1, 1}) + eval(Acc{2, 0}));
    }
    template <typename Evaluation>
    GT_FUNCTION static float_type do01(Evaluation &eval) {
        return 4. * eval(Acc{0, 1}) - (eval(Acc{-1, 1}) + eval(Acc{0, 0}) + eval(Acc{0, 2}) + eval(Acc{1, 1}));
    }
};

template <typename Acc>
struct lap<variation::call, Acc> {
    template <typename Evaluation>
    GT_FUNCTION static float_type do00(Evaluation &eval) {
        return call<lap_function>::at<0, 0, 0>::with(eval, Acc{});
    }
    template <typename Evaluation>
    GT_FUNCTION static float_type do10(Evaluation &eval) {
        return call<lap_function>::at<1, 0, 0>::with(eval, Acc{});
    }
    template <typename Evaluation>
    GT_FUNCTION static float_type do01(Evaluation &eval) {
        return call<lap_function>::at<0, 1, 0>::with(eval, Acc{});
    }
};

template <typename Acc>
struct lap<variation::call_offsets, Acc> {
    template <typename Evaluation>
    GT_FUNCTION static float_type do00(Evaluation &eval) {
        return call<lap_function>::with(eval, Acc{0, 0});
    }
    template <typename Evaluation>
    GT_FUNCTION static float_type do10(Evaluation &eval) {
        return call<lap_function>::with(eval, Acc{1, 0});
    }
    template <typename Evaluation>
    GT_FUNCTION static float_type do01(Evaluation &eval) {
        return call<lap_function>::with(eval, Acc{0, 1});
    }
};

template <typename Acc>
struct lap<variation::procedures, Acc> {
    template <typename Evaluation>
    GT_FUNCTION static float_type do00(Evaluation &eval) {
        float_type res;
        call_proc<lap_function>::at<0, 0, 0>::with(eval, res, Acc{});
        return res;
    }
    template <typename Evaluation>
    GT_FUNCTION static float_type do10(Evaluation &eval) {
        float_type res;
        call_proc<lap_function>::at<1, 0, 0>::with(eval, res, Acc{});
        return res;
    }
    template <typename Evaluation>
    GT_FUNCTION static float_type do01(Evaluation &eval) {
        float_type res;
        call_proc<lap_function>::at<0, 1, 0>::with(eval, res, Acc{});
        return res;
    }
};

template <typename Acc>
struct lap<variation::procedures_offsets, Acc> {
    template <typename Evaluation>
    GT_FUNCTION static float_type do00(Evaluation &eval) {
        float_type res;
        call_proc<lap_function>::with(eval, res, Acc{0, 0});
        return res;
    }
    template <typename Evaluation>
    GT_FUNCTION static float_type do10(Evaluation &eval) {
        float_type res;
        call_proc<lap_function>::with(eval, res, Acc{1, 0});
        return res;
    }
    template <typename Evaluation>
    GT_FUNCTION static float_type do01(Evaluation &eval) {
        float_type res;
        call_proc<lap_function>::with(eval, res, Acc{0, 1});
        return res;
    }
};

template <variation Variation>
struct flx_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 2, -1, 1>>;

    using param_list = make_param_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = lap<Variation, in>::do10(eval) - lap<Variation, in>::do00(eval);
        eval(out()) = eval(out()) * (eval(in(1, 0)) - eval(in(0, 0))) > 0 ? 0.0 : eval(out());
    }
};

template <variation Variation>
struct fly_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 2>>;

    using param_list = make_param_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = lap<Variation, in>::do01(eval) - lap<Variation, in>::do00(eval);
        eval(out()) = eval(out()) * (eval(in(0, 1, 0)) - eval(in(0, 0, 0))) > 0 ? 0.0 : eval(out());
    }
};

struct out_function {
    using out = inout_accessor<0>;
    using in = in_accessor<1>;
    using flx = in_accessor<2, extent<-1, 0, 0, 0>>;
    using fly = in_accessor<3, extent<0, 0, -1, 0>>;
    using coeff = in_accessor<4>;

    using param_list = make_param_list<out, in, flx, fly, coeff>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out()) = eval(in()) - eval(coeff()) * (eval(flx()) - eval(flx(-1, 0)) + eval(fly()) - eval(fly(0, -1)));
    }
};

struct horizontal_diffusion_functions : regression_fixture<2> {
    tmp_arg<0> p_flx;
    tmp_arg<1> p_fly;
    arg<1> p_coeff;
    arg<2> p_in;
    arg<3> p_out;

    template <variation Variation>
    void do_test() {
        auto out = make_storage();

        horizontal_diffusion_repository repo(d1(), d2(), d3());

        make_computation(p_in = make_storage(repo.in),
            p_out = out,
            p_coeff = make_storage(repo.coeff),
            make_multistage(execute::forward(),
                define_caches(cache<cache_type::ij, cache_io_policy::local>(p_flx, p_fly)),
                make_independent(
                    make_stage<flx_function<Variation>>(p_flx, p_in), make_stage<fly_function<Variation>>(p_fly, p_in)),
                make_stage<out_function>(p_out, p_in, p_flx, p_fly, p_coeff)))
            .run();

        verify(make_storage(repo.out), out);
    }
};

TEST_F(horizontal_diffusion_functions, monolithic) { do_test<variation::monolithic>(); }

TEST_F(horizontal_diffusion_functions, call) { do_test<variation::call>(); }

TEST_F(horizontal_diffusion_functions, call_offsets) { do_test<variation::call_offsets>(); }

TEST_F(horizontal_diffusion_functions, procedures) { do_test<variation::procedures>(); }

TEST_F(horizontal_diffusion_functions, procedures_offsets) { do_test<variation::procedures_offsets>(); }
