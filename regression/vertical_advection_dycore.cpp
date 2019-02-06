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

#include "vertical_advection_defs.hpp"
#include "vertical_advection_repository.hpp"

/*
  This file shows an implementation of the "vertical advection" stencil used in COSMO for U field
 */

using namespace gridtools;

// This is the definition of the special regions in the "vertical" direction
using axis_t = axis<1>::with_offset_limit<3>;
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

    using arg_list = make_arg_list<utens_stage, wcon, u_stage, u_pos, utens, dtr_stage, acol, bcol, ccol, dcol>;

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

    using arg_list = make_arg_list<utens_stage, u_pos, dtr_stage, ccol, dcol, data_col>;

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

struct vertical_advection_dycore : regression_fixture<3, axis_t> {
    arg<0> p_utens_stage;
    arg<1> p_u_stage;
    arg<2> p_wcon;
    arg<3> p_u_pos;
    arg<4> p_utens;
    arg<5, scalar_storage_type> p_dtr_stage;
    tmp_arg<0> p_acol;
    tmp_arg<1> p_bcol;
    tmp_arg<2> p_ccol;
    tmp_arg<3> p_dcol;
    tmp_arg<4> p_data_col;

    vertical_advection_repository repo{d1(), d2(), d3()};

    storage_type utens_stage = make_storage(repo.utens_stage_in);

    void verify_utens_stage() { verify(make_storage(repo.utens_stage_out), utens_stage); }
};

TEST_F(vertical_advection_dycore, test) {
    auto comp = make_computation(p_utens_stage = utens_stage,
        p_u_stage = make_storage(repo.u_stage),
        p_wcon = make_storage(repo.wcon),
        p_u_pos = make_storage(repo.u_pos),
        p_utens = make_storage(repo.utens),
        p_dtr_stage = make_storage<scalar_storage_type>(repo.dtr_stage),
        make_multistage(enumtype::execute<enumtype::forward>(),
            define_caches(cache<K, cache_io_policy::local>(p_acol),
                cache<K, cache_io_policy::local>(p_bcol),
                cache<K, cache_io_policy::flush>(p_ccol),
                cache<K, cache_io_policy::flush>(p_dcol),
                cache<K, cache_io_policy::fill>(p_u_stage)),
            make_stage<u_forward_function>(
                p_utens_stage, p_wcon, p_u_stage, p_u_pos, p_utens, p_dtr_stage, p_acol, p_bcol, p_ccol, p_dcol)),
        make_multistage(enumtype::execute<enumtype::backward>(),
            define_caches(cache<K, cache_io_policy::local>(p_data_col)),
            make_stage<u_backward_function>(p_utens_stage, p_u_pos, p_dtr_stage, p_ccol, p_dcol, p_data_col)));
    comp.run();
    verify_utens_stage();
    benchmark(comp);
}

TEST_F(vertical_advection_dycore, with_extents) {
    make_computation(p_utens_stage = utens_stage,
        p_u_stage = make_storage(repo.u_stage),
        p_wcon = make_storage(repo.wcon),
        p_u_pos = make_storage(repo.u_pos),
        p_utens = make_storage(repo.utens),
        p_dtr_stage = make_storage<scalar_storage_type>(repo.dtr_stage),
        make_multistage(enumtype::execute<enumtype::forward>(),
            define_caches(cache<K, cache_io_policy::local>(p_acol),
                cache<K, cache_io_policy::local>(p_bcol),
                cache<K, cache_io_policy::flush>(p_ccol),
                cache<K, cache_io_policy::flush>(p_dcol),
                cache<K, cache_io_policy::fill>(p_u_stage)),
            make_stage_with_extent<u_forward_function, extent<>>(
                p_utens_stage, p_wcon, p_u_stage, p_u_pos, p_utens, p_dtr_stage, p_acol, p_bcol, p_ccol, p_dcol)),
        make_multistage(enumtype::execute<enumtype::backward>(),
            define_caches(cache<K, cache_io_policy::local>(p_data_col)),
            make_stage_with_extent<u_backward_function, extent<>>(
                p_utens_stage, p_u_pos, p_dtr_stage, p_ccol, p_dcol, p_data_col)))
        .run();
    verify_utens_stage();
}
