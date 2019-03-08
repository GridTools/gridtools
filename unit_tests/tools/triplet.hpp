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
struct triplet {
    int a = 0, b = 0, c = 0;

    GT_DECLARE_DEFAULT_EMPTY_CTOR(triplet);

    GT_FUNCTION
    triplet(int a, int b, int c) : a(a), b(b), c(c) {}

    GT_FUNCTION
    bool operator==(triplet other) const { return (a == other.a) and (b == other.b) and (c == other.c); }

    GT_FUNCTION
    bool operator!=(triplet other) const { return not(*this == other); }
};

std::ostream &operator<<(std::ostream &s, triplet t) { return s << "[" << t.a << " " << t.b << " " << t.c << "]"; }
