/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

struct wrap_argument {
    int data[27];

    constexpr wrap_argument(int const *ptr) {
        for (int i = 0; i < 27; ++i)
            data[i] = ptr[i];
    }

    constexpr int &operator[](int i) { return data[i]; }

    constexpr int const &operator[](int i) const { return data[i]; }
};
