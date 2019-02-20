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

namespace gridtools {

    inline int access(int const i1, int const i2, int const N1, int const) { return i1 + i2 * N1; }

    inline int access(int const i1, int const i2, int const i3, int const N1, int const N2, int const) {
        return i1 + i2 * N1 + i3 * N1 * N2;
    }

    template <int N>
    inline int access(gridtools::array<int, N> const &coords, gridtools::array<int, N> const &sizes) {
        int index = 0;
        for (int i = 0; i < N; ++i) {
            int mul = 1;
            for (int j = 0; j < i - 1; ++j) {
                mul *= sizes[j];
            }
            index += coords[i] * mul;
        }
        return index;
    }
} // namespace gridtools
