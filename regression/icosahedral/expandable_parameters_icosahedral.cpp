/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtest/gtest.h>

#include <gridtools/stencil-composition/stencil-composition.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;

template <uint_t>
struct functor_copy {
    using out = inout_accessor<0, enumtype::cells>;
    using in = in_accessor<1, enumtype::cells>;
    using param_list = make_param_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval) {
        eval(out{}) = eval(in{});
    }
};

using expandable_parameters_icosahedral = regression_fixture<>;

TEST_F(expandable_parameters_icosahedral, test) {
    using storages_t = std::vector<storage_type<cells>>;
    storages_t in = {make_storage<cells>(10.),
        make_storage<cells>(20.),
        make_storage<cells>(30.),
        make_storage<cells>(40.),
        make_storage<cells>(50.)};
    storages_t out = {make_storage<cells>(1.),
        make_storage<cells>(2.),
        make_storage<cells>(3.),
        make_storage<cells>(4.),
        make_storage<cells>(5.)};

    arg<0, cells, storages_t> p_out;
    arg<1, cells, storages_t> p_in;

    gridtools::make_computation<backend_t>(expand_factor<2>(),
        make_grid(),
        p_out = out,
        p_in = in,
        make_multistage(execute::forward(), make_stage<functor_copy, topology_t, cells>(p_out, p_in)))
        .run();

    for (size_t i = 0; i != in.size(); ++i)
        verify(in[i], out[i]);
}
