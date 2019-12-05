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

#include <gridtools/stencil_composition/icosahedral.hpp>
#include <gridtools/tools/regression_fixture.hpp>

using namespace gridtools;
using namespace icosahedral;

struct functor_copy {
    using out = inout_accessor<0, cells>;
    using in = in_accessor<1, cells>;
    using param_list = make_param_list<out, in>;
    using location = cells;

    template <class Eval>
    GT_FUNCTION static void apply(Eval &&eval) {
        eval(out()) = eval(in());
    }
};

using expandable_parameters_icosahedral = regression_fixture<>;

TEST_F(expandable_parameters_icosahedral, test) {
    using storages_t = std::vector<decltype(make_storage<cells>())>;
    storages_t out = {make_storage<cells>(),
        make_storage<cells>(),
        make_storage<cells>(),
        make_storage<cells>(),
        make_storage<cells>()};
    expandable_run<2>([](auto out, auto in) { return execute_parallel().stage(functor_copy(), out, in); },
        backend_t(),
        make_grid(),
        out,
        storages_t{make_storage<cells>(10),
            make_storage<cells>(20),
            make_storage<cells>(30),
            make_storage<cells>(40),
            make_storage<cells>(50)});
    for (size_t i = 0; i != out.size(); ++i)
        verify((i + 1) * 10, out[i]);
}
