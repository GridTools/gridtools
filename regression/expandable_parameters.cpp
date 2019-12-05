/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <vector>

#include <gtest/gtest.h>

#include <gridtools/stencil_composition/cartesian.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;
using namespace cartesian;

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

    expandable_run<2>(
        [](auto in, auto out) {
            GT_DECLARE_EXPANDABLE_TMP(float_type, tmp);
            return execute_parallel().ij_cached(tmp).stage(copy_functor(), tmp, in).stage(copy_functor(), out, tmp);
        },
        backend_t(),
        make_grid(),
        in,
        out);

    for (size_t i = 0; i != in.size(); ++i)
        verify(in[i], out[i]);
}
