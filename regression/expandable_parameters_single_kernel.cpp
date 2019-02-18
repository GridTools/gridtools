/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
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

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;

struct functor_single_kernel {
    using parameters1_out = inout_accessor<0>;
    using parameters2_out = inout_accessor<1>;
    using parameters3_out = inout_accessor<2>;
    using parameters4_out = inout_accessor<3>;
    using parameters5_out = inout_accessor<4>;

    using parameters1_in = in_accessor<5>;
    using parameters2_in = in_accessor<6>;
    using parameters3_in = in_accessor<7>;
    using parameters4_in = in_accessor<8>;
    using parameters5_in = in_accessor<9>;

    using param_list = make_param_list<parameters1_out,
        parameters2_out,
        parameters3_out,
        parameters4_out,
        parameters5_out,
        parameters1_in,
        parameters2_in,
        parameters3_in,
        parameters4_in,
        parameters5_in>;

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
