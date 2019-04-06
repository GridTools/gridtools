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
#include <gridtools/tools/backend_select.hpp>

using namespace gridtools;
using namespace execute;

struct TensionShearFunction {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(T_sqr_s),
        GT_INOUT_ACCESSOR(S_sqr_uv),
        GT_IN_ACCESSOR(u_in, extent<-1, 0, 0, 1>),
        GT_IN_ACCESSOR(v_in, extent<0, 1, -1, 0>));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {}
};

struct SmagCoeffFunction {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(smag_u),
        GT_INOUT_ACCESSOR(smag_v),
        GT_IN_ACCESSOR(T_sqr_s, extent<0, 1, 0, 1>),
        GT_IN_ACCESSOR(S_sqr_uv, extent<-1, 0, -1, 0>));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {}
};

struct SmagUpdateFunction {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(u_out),
        GT_INOUT_ACCESSOR(v_out),
        GT_IN_ACCESSOR(u_in, extent<-1, 1, -1, 1>),
        GT_IN_ACCESSOR(v_in, extent<-1, 1, -1, 1>),
        GT_IN_ACCESSOR(smag_u),
        GT_IN_ACCESSOR(smag_v));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {}
};

TEST(multiple_outputs, compute_extents) {

    typedef storage_traits<backend_t>::storage_info_t<0, 3> meta_data_t;
    typedef storage_traits<backend_t>::data_store_t<float_type, meta_data_t> storage_t;

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
