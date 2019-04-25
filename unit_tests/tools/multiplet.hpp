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

#include <iosfwd>
#include <ostream>

#include <gridtools/common/defs.hpp>
#include <gridtools/common/host_device.hpp>

/**
   @brief Small value type to use in tests where we want to check the
   values in a fields, for instance to check if layouts works, on in
   communication tests
*/

template <gridtools::uint_t N>
struct multiplet {
    int data[N];

    GT_FUNCTION
    bool operator==(multiplet const &other) const {
        for (size_t i = 0; i < N; ++i) {
            if (data[i] != other.data[i])
                return false;
        }
        return true;
    }

    GT_FUNCTION
    bool operator!=(multiplet const &other) const { return not(*this == other); }

    friend std::ostream &operator<<(std::ostream &s, multiplet t) {
        bool need_comma = false;
        s << "[";
        for (size_t i = 0; i < N; ++i) {
            if (need_comma) {
                s << ",";
            }
            need_comma = true;
            s << t.data[i];
        }
        s << "]";
        return s;
    }
};

using triplet = multiplet<3>;
