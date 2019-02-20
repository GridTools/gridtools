/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include "timer.hpp"
#include <limits>
#include <string>

namespace gridtools {

    /**
     * @class timer_dummy
     * Dummy timer implementation doing nothing in order to avoid runtime overhead
     */
    class timer_dummy : public timer<timer_dummy> // CRTP
    {
      public:
        GT_FUNCTION_HOST timer_dummy(std::string name) : timer<timer_dummy>(name) {}

        GT_FUNCTION_HOST void set_impl(double const & /*time_*/) {}

        GT_FUNCTION_HOST void start_impl() {}

        GT_FUNCTION_HOST double pause_impl() { return std::numeric_limits<double>::quiet_NaN(); }
    };
} // namespace gridtools
