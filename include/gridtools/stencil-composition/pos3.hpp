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

#include "../common/host_device.hpp"

namespace gridtools {
    template <class T>
    struct pos3 {
        T i, j, k;
    };

    template <class T>
    GT_FUNCTION constexpr pos3<T> make_pos3(T const &i, T const &j, T const &k) {
        return {i, j, k};
    }
} // namespace gridtools
