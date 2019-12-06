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

#include <gridtools/stencil_composition/cartesian.hpp>
#include <gridtools/stencil_composition/global_parameter.hpp>
#include <gridtools/tools/cartesian_regression_fixture.hpp>

#include "vertical_advection_defs.hpp"
#include "vertical_advection_repository.hpp"

/*
  This file shows an implementation of the "vertical advection" stencil used in COSMO for U field
  */

using namespace gridtools;
using namespace cartesian;

// This is the definition of the special regions in the "vertical" direction
using axis_t = axis<1, axis_config::offset_limit<3>>;
using full_t = axis_t::full_interval;

struct u_forward_function {
    using utens_stage = in_accessor<0>;
    using wcon = in_accessor<1, extent<0, 1, 0, 0, 0, 1>>;
    using u_stage = in_accessor<2, extent<0, 0, 0, 0, -1, 1>>;
    using u_pos = in_accessor<3>;
    using utens = in_accessor<4>;
    using dtr_stage = in_accessor<5>;
    using acol = inout_accessor<6>;
    using bcol = inout_accessor<7>;
    using ccol = inout_accessor<8, extent<0, 0, 0, 0, -1, 0>>;
    using dcol = inout_accessor<9, extent<0, 0, 0, 0, -1, 0>>;

    using param_list = make_param_list<utens_stage, wcon, u_stage, u_pos, utens, dtr_stage, acol, bcol, ccol, dcol>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1> interval) {
        // TODO use Average function here
        float_type gav = -float_type{.25} * (eval(wcon(1, 0, 0)) + eval(wcon(0, 0, 0)));
        float_type gcv = float_type{.25} * (eval(wcon(1, 0, 1)) + eval(wcon(0, 0, 1)));

        float_type as = gav * BET_M;
        float_type cs = gcv * BET_M;

        eval(acol()) = gav * BET_P;
        eval(ccol()) = gcv * BET_P;
        eval(bcol()) = eval(dtr_stage()) - eval(acol()) - eval(ccol());

        float_type correctionTerm =
            -as * (eval(u_stage(0, 0, -1)) - eval(u_stage())) - cs * (eval(u_stage(0, 0, 1)) - eval(u_stage()));
        // update the d column
        compute_d_column(eval, correctionTerm);
        thomas_forward(eval, interval);
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level interval) {
        float_type gav = -float_type{.25} * (eval(wcon(1, 0, 0)) + eval(wcon()));
        float_type as = gav * BET_M;

        eval(acol()) = gav * BET_P;
        eval(bcol()) = eval(dtr_stage()) - eval(acol());

        float_type correctionTerm = -as * (eval(u_stage(0, 0, -1)) - eval(u_stage()));

        // update the d column
        compute_d_column(eval, correctionTerm);
        thomas_forward(eval, interval);
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::first_level interval) {
        float_type gcv = float_type{.25} * (eval(wcon(1, 0, 1)) + eval(wcon(0, 0, 1)));
        float_type cs = gcv * BET_M;

        eval(ccol()) = gcv * BET_P;
        eval(bcol()) = eval(dtr_stage()) - eval(ccol());

        float_type correctionTerm = -cs * (eval(u_stage(0, 0, 1)) - eval(u_stage()));
        // update the d column
        compute_d_column(eval, correctionTerm);
        thomas_forward(eval, interval);
    }

  private:
    template <typename Evaluation>
    GT_FUNCTION static void compute_d_column(Evaluation &eval, float_type correctionTerm) {
        eval(dcol()) = eval(dtr_stage()) * eval(u_pos()) + eval(utens()) + eval(utens_stage()) + correctionTerm;
    }

    template <typename Evaluation>
    GT_FUNCTION static void thomas_forward(Evaluation &eval, full_t::modify<1, -1>) {
        float_type divided = float_type{1} / (eval(bcol()) - (eval(ccol(0, 0, -1)) * eval(acol())));
        eval(ccol()) = eval(ccol()) * divided;
        eval(dcol()) = (eval(dcol()) - (eval(dcol(0, 0, -1)) * eval(acol()))) * divided;
    }

    template <typename Evaluation>
    GT_FUNCTION static void thomas_forward(Evaluation &eval, full_t::last_level) {
        float_type divided = float_type{1} / (eval(bcol()) - eval(ccol(0, 0, -1)) * eval(acol()));
        eval(dcol()) = (eval(dcol()) - eval(dcol(0, 0, -1)) * eval(acol())) * divided;
    }

    template <typename Evaluation>
    GT_FUNCTION static void thomas_forward(Evaluation &eval, full_t::first_level) {
        float_type divided = float_type{1} / eval(bcol());
        eval(ccol()) = eval(ccol()) * divided;
        eval(dcol()) = eval(dcol()) * divided;
    }
};

struct u_backward_function {
    using utens_stage = inout_accessor<0>;
    using u_pos = in_accessor<1>;
    using dtr_stage = in_accessor<2>;
    using ccol = in_accessor<3>;
    using dcol = in_accessor<4>;
    using data_col = inout_accessor<5, extent<0, 0, 0, 0, 0, 1>>;

    using param_list = make_param_list<utens_stage, u_pos, dtr_stage, ccol, dcol, data_col>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, full_t::modify<0, -1> interval) {
        eval(utens_stage()) = eval(dtr_stage()) * (thomas_backward(eval, interval) - eval(u_pos()));
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval, full_t::last_level interval) {
        eval(utens_stage()) = eval(dtr_stage()) * (thomas_backward(eval, interval) - eval(u_pos()));
    }

  private:
    template <typename Evaluation>
    GT_FUNCTION static float_type thomas_backward(Evaluation &eval, full_t::modify<0, -1>) {
        float_type datacol = eval(dcol()) - eval(ccol()) * eval(data_col(0, 0, 1));
        eval(data_col()) = datacol;
        return datacol;
    }

    template <typename Evaluation>
    GT_FUNCTION static float_type thomas_backward(Evaluation &eval, full_t::last_level) {
        float_type datacol = eval(dcol());
        eval(data_col()) = datacol;
        return datacol;
    }
};

const auto vertical_advection = [](auto utens_stage, auto u_stage, auto wcon, auto u_pos, auto utens, auto dtr_stage) {
    GT_DECLARE_TMP(float_type, acol, bcol, ccol, dcol, data_col);
    return multi_pass(
        execute_forward()
            .k_cached(acol, bcol)
            .k_cached(cache_io_policy::flush(), ccol, dcol)
            .k_cached(cache_io_policy::fill(), u_stage)
            .stage(u_forward_function(), utens_stage, wcon, u_stage, u_pos, utens, dtr_stage, acol, bcol, ccol, dcol),
        execute_backward().k_cached(data_col).stage(
            u_backward_function(), utens_stage, u_pos, dtr_stage, ccol, dcol, data_col));
};

#if defined(GT_BACKEND_CUDA)
using modified_backend_t = cuda::backend<integral_constant<int_t, 256>, integral_constant<int_t, 1>>;
#else
using modified_backend_t = backend_t;
#endif

using vertical_advection_dycore = regression_fixture<3, axis_t>;

TEST_F(vertical_advection_dycore, test) {
    vertical_advection_repository repo{d(0), d(1), d(2)};
    storage_type utens_stage = make_storage(repo.utens_stage_in);
    auto comp = [grid = make_grid(),
                    &utens_stage,
                    u_stage = make_storage(repo.u_stage),
                    wcon = make_storage(repo.wcon),
                    u_pos = make_storage(repo.u_pos),
                    utens = make_storage(repo.utens),
                    dtr_stage = make_global_parameter((float_type)repo.dtr_stage)] {
        run(vertical_advection, modified_backend_t(), grid, utens_stage, u_stage, wcon, u_pos, utens, dtr_stage);
    };
    comp();
    verify(repo.utens_stage_out, utens_stage);
    benchmark(comp);
}
