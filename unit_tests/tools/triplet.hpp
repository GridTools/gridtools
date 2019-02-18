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

#include <gridtools/common/host_device.hpp>
#include <iosfwd>
#include <ostream>

/**
   @brief Small value type to use in tests where we want to check the
   values in a fields, for instance to check if layouts works, on in
   communication tests
*/
struct triplet {
    int a = 0, b = 0, c = 0;

#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ < 9)
    GT_FUNCTION triplet() {}
#else
    triplet() = default;
#endif

    GT_FUNCTION
    triplet(int a, int b, int c) : a(a), b(b), c(c) {}

    GT_FUNCTION
    bool operator==(triplet other) const { return (a == other.a) and (b == other.b) and (c == other.c); }

    GT_FUNCTION
    bool operator!=(triplet other) const { return not(*this == other); }
};

std::ostream &operator<<(std::ostream &s, triplet t) { return s << "[" << t.a << " " << t.b << " " << t.c << "]"; }
