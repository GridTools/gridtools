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

#include "defs.hpp"

namespace gridtools {
    constexpr int static_pow3(uint_t i) { return i ? 3 * static_pow3(i - 1) : 1; }
} // namespace gridtools
