/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/*
 * test_computation.cpp
 *
 *  Created on: Mar 9, 2015
 *      Author: carlosos
 */

#include <boost/fusion/include/make_vector.hpp>
#include <boost/mpl/equal.hpp>

#include "gtest/gtest.h"

#include <gridtools/stencil_composition/backend.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>

using namespace gridtools;

namespace make_computation_test {
    constexpr int level_offset_limit = 1;

    template <uint_t Splitter, int_t Offset>
    using level_t = level<Splitter, Offset, level_offset_limit>;

    typedef gridtools::interval<level_t<0, -1>, level_t<1, -1>> axis;
    using backend_t = backend<target::x86, strategy::block>;
    using icosahedral_topology_t = gridtools::icosahedral_topology<backend_t>;

    struct test_functor {
        using in = in_accessor<0, icosahedral_topology_t::cells, extent<1>>;
        using param_list = make_param_list<in>;

        template <typename Evaluation>
        GT_FUNCTION static void apply(Evaluation &eval, axis) {}
    };
} // namespace make_computation_test

TEST(MakeComputation, Basic) {}
