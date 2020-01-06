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

namespace gridtools {
    template <class T>
    struct pos3 {
        T i, j, k;
    };

    template <class T>
    constexpr pos3<T> make_pos3(T i, T j, T k) {
        return {i, j, k};
    }
} // namespace gridtools
