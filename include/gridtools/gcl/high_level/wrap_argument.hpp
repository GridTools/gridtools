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

    constexpr wrap_argument(int const *ptr)
        : data{ptr[0],
              ptr[1],
              ptr[2],
              ptr[3],
              ptr[4],
              ptr[5],
              ptr[6],
              ptr[7],
              ptr[8],
              ptr[9],
              ptr[10],
              ptr[11],
              ptr[12],
              ptr[13],
              ptr[14],
              ptr[15],
              ptr[16],
              ptr[17],
              ptr[18],
              ptr[19],
              ptr[20],
              ptr[21],
              ptr[22],
              ptr[23],
              ptr[24],
              ptr[25],
              ptr[26]} {}

    constexpr int &operator[](int i) { return data[i]; }

    constexpr int const &operator[](int i) const { return data[i]; }
};
