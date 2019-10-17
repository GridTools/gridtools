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

#include <limits>

namespace gridtools {
    /**
     * @class timer_dummy
     * Dummy timer implementation doing nothing in order to avoid runtime overhead
     */
    struct timer_dummy {
        void start_impl() {}
        double pause_impl() { return std::numeric_limits<double>::quiet_NaN(); }
    };
} // namespace gridtools
