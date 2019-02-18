/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "../../meta/id.hpp"
#include "../host_device.hpp"

namespace gridtools {
    /**
     * `boost::implicit_cast` clone with constexpr and target specifiers
     *
     * The use of identity creates a non-deduced form, so that the explicit template argument must be supplied
     */
    template <class T>
    GT_FUNCTION constexpr T implicit_cast(typename meta::lazy::id<T>::type x) {
        return x;
    }
} // namespace gridtools
