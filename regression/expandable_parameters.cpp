/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <vector>

#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;

struct copy_functor {
    using parameters_out = inout_accessor<0>;
    using parameters_in = accessor<1>;

    using param_list = make_param_list<parameters_out, parameters_in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation &eval) {
        eval(parameters_out{}) = eval(parameters_in{});
    }
};

using expandable_parameters = regression_fixture<>;

TEST_F(expandable_parameters, test) {
    using storages_t = std::vector<storage_type>;
    storages_t out = {make_storage(1.), make_storage(2.), make_storage(3.), make_storage(4.), make_storage(5.)};
    storages_t in = {make_storage(-1.), make_storage(-2.), make_storage(-3.), make_storage(-4.), make_storage(-5.)};

    arg<0, storages_t> p_out;
    arg<1, storages_t> p_in;
    tmp_arg<2, storages_t> p_tmp;

    gridtools::make_computation<backend_t>(expand_factor<2>(),
        make_grid(),
        p_out = out,
        p_in = in,
        make_multistage(execute::forward(),
            define_caches(cache<cache_type::ij, cache_io_policy::local>(p_tmp)),
            make_stage<copy_functor>(p_tmp, p_in),
            make_stage<copy_functor>(p_out, p_tmp)))
        .run();
    for (size_t i = 0; i != in.size(); ++i)
        verify(in[i], out[i]);
}
