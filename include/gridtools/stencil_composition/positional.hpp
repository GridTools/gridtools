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

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../common/hymap.hpp"

namespace gridtools {

    // Represents position in the computation space.
    // Models SID concept
    template <class Dim>
    class positional {
        int_t m_val;

        struct stride {};

        friend GT_FUNCTION positional operator+(positional lhs, positional rhs) { return {lhs.m_val + rhs.m_val}; }

        friend GT_FUNCTION void sid_shift(positional &p, stride, int_t offset) { p.m_val += offset; }
        friend positional sid_get_origin(positional obj) { return obj; }
        friend typename hymap::keys<Dim>::template values<stride> sid_get_strides(positional) { return {}; }

      public:
        GT_FUNCTION positional(int_t val = 0) : m_val{val} {}

        GT_FUNCTION int operator*() const { return m_val; }
        GT_FUNCTION positional const &operator()() const { return *this; }
    };

    template <class Dim>
    positional<Dim> sid_get_ptr_diff(positional<Dim>);
} // namespace gridtools
