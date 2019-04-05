/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#undef FUSION_MAX_VECTOR_SIZE
#undef FUSION_MAX_MAP_SIZE
#define FUSION_MAX_VECTOR_SIZE 40
#define FUSION_MAX_MAP_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_LIMIT_VECTOR_SIZE FUSION_MAX_VECTOR_SIZE
#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#include <vector>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;

struct functor_single_kernel {
    GT_DEFINE_ACCESSORS(GT_INOUT_ACCESSOR(parameters1_out),
        GT_INOUT_ACCESSOR(parameters2_out),
        GT_INOUT_ACCESSOR(parameters3_out),
        GT_INOUT_ACCESSOR(parameters4_out),
        GT_INOUT_ACCESSOR(parameters5_out),
        GT_IN_ACCESSOR(parameters1_in),
        GT_IN_ACCESSOR(parameters2_in),
        GT_IN_ACCESSOR(parameters3_in),
        GT_IN_ACCESSOR(parameters4_in),
        GT_IN_ACCESSOR(parameters5_in));

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(parameters1_out()) = eval(parameters1_in());
        eval(parameters2_out()) = eval(parameters2_in());
        eval(parameters3_out()) = eval(parameters3_in());
        eval(parameters4_out()) = eval(parameters4_in());
        eval(parameters5_out()) = eval(parameters5_in());
    }
};

using expandable_parameters_single_kernel = regression_fixture<>;

TEST_F(expandable_parameters_single_kernel, test) {
    std::vector<storage_type> out = {
        make_storage(1.), make_storage(2.), make_storage(3.), make_storage(4.), make_storage(5.)};
    std::vector<storage_type> in = {
        make_storage(-1.), make_storage(-2.), make_storage(-3.), make_storage(-4.), make_storage(-5.)};

    make_computation(p_0 = out[0],
        p_1 = out[1],
        p_2 = out[2],
        p_3 = out[3],
        p_4 = out[4],
        p_5 = in[0],
        p_6 = in[1],
        p_7 = in[2],
        p_8 = in[3],
        p_9 = in[4],
        make_multistage(execute::forward(),
            define_caches(cache<cache_type::ij, cache_io_policy::local>(p_tmp_0, p_tmp_1, p_tmp_2, p_tmp_3, p_tmp_4)),
            make_stage<functor_single_kernel>(p_tmp_0, p_tmp_1, p_tmp_2, p_tmp_3, p_tmp_4, p_5, p_6, p_7, p_8, p_9),
            make_stage<functor_single_kernel>(p_0, p_1, p_2, p_3, p_4, p_tmp_0, p_tmp_1, p_tmp_2, p_tmp_3, p_tmp_4)))
        .run();

    for (size_t i = 0; i != in.size(); ++i)
        verify(in[i], out[i]);
}
