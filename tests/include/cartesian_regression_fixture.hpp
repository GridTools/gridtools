/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "cartesian_fixture.hpp"
#include "regression_fixture.hpp"

namespace gridtools {
    namespace cartesian {
        template <size_t Halo = 0, class Axis = axis<1>>
        struct regression_fixture : regression_fixture_templ<computation_fixture<Halo, Axis>> {};
    } // namespace cartesian
} // namespace gridtools
