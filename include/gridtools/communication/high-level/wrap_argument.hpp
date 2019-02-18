/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

struct wrap_argument {
    int data[27];

    GT_FUNCTION wrap_argument(int const *ptr) {
        for (int i = 0; i < 27; ++i)
            data[i] = ptr[i];
    }

    GT_FUNCTION int &operator[](int i) { return data[i]; }

    GT_FUNCTION int const &operator[](int i) const { return data[i]; }
};
