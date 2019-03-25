/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "gtest/gtest.h"
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>

using namespace gridtools;
struct func {
    using p1 = accessor<0, intent::in>;
    using p2 = accessor<1, intent::inout>;
    using param_list = make_param_list<p1, p2>;

    template <typename Evaluation>
    void apply(Evaluation &eval) {}
};

struct func_call {
    using p1 = accessor<0, intent::in>;
    using p2 = accessor<1, intent::inout>;
    using param_list = make_param_list<p1, p2>;

    template <typename Evaluation>
    void apply(Evaluation &eval) {
        call<func>::with(eval);
    }
};

struct storage_stub {
    using iterator = void;
    using value_type = void;
};

TEST(default_interval, test) {

    make_stage<func>(arg<0, storage_stub>(), arg<1, storage_stub>());
    make_stage<func_call>(arg<0, storage_stub>(), arg<1, storage_stub>());
}
