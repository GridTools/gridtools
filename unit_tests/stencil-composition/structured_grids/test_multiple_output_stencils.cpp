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

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;
using namespace execute;

struct TensionShearFunction {
    using T_sqr_s = inout_accessor<0>;
    using S_sqr_uv = inout_accessor<1>;

    using u_in = in_accessor<2, extent<-1, 0, 0, 1>>;
    using v_in = in_accessor<3, extent<0, 1, -1, 0>>;

    using param_list = make_param_list<T_sqr_s, S_sqr_uv, u_in, v_in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {}
};

struct SmagCoeffFunction {
    using smag_u = inout_accessor<0>;
    using smag_v = inout_accessor<1>;

    using T_sqr_s = in_accessor<2, extent<0, 1, 0, 1>>;
    using S_sqr_uv = in_accessor<3, extent<-1, 0, -1, 0>>;

    using param_list = make_param_list<smag_u, smag_v, T_sqr_s, S_sqr_uv>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {}
};

struct SmagUpdateFunction {
    using u_out = inout_accessor<0>;
    using v_out = inout_accessor<1>;

    using u_in = in_accessor<2, extent<-1, 1, -1, 1>>;
    using v_in = in_accessor<3, extent<-1, 1, -1, 1>>;
    using smag_u = in_accessor<4>;
    using smag_v = in_accessor<5>;

    using param_list = make_param_list<u_out, v_out, u_in, v_in, smag_u, smag_v>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {}
};

TEST(multiple_outputs, compute_extents) {

    typedef backend_t::storage_traits_t::storage_info_t<0, 3> meta_data_t;
    typedef backend_t::storage_traits_t::data_store_t<float_type, meta_data_t> storage_t;

    using T_sqr_s = tmp_arg<0, storage_t>;
    using S_sqr_uv = tmp_arg<1, storage_t>;
    using smag_u = tmp_arg<2, storage_t>;
    using smag_v = tmp_arg<3, storage_t>;

    // Output fields
    using u_out = arg<4, storage_t>;
    using v_out = arg<5, storage_t>;

    // Input fields
    using u_in = arg<6, storage_t>;
    using v_in = arg<7, storage_t>;

    halo_descriptor di{2, 2, 2, 7, 10};
    halo_descriptor dj{2, 2, 2, 7, 10};
    auto grid_ = make_grid(di, dj, 10);

    make_computation<backend_t>(grid_,
        make_multistage(execute::forward(),
            make_stage<TensionShearFunction>(T_sqr_s(), S_sqr_uv(), u_in(), v_in()),
            make_stage<SmagCoeffFunction>(smag_u(), smag_v(), T_sqr_s(), S_sqr_uv()),
            make_stage<SmagUpdateFunction>(u_out(), v_out(), u_in(), v_in(), smag_u(), smag_v())));
}
