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
#include "dim.hpp"
#include "sid/concept.hpp"

namespace gridtools {

    // Represents position in the computation space.
    // Models SID concept
    struct positional {
        int_t i;
        int_t j;
        int_t k;

        GT_FUNCTION positional const &operator*() const { return *this; }
        GT_FUNCTION positional const &operator()() const { return *this; }

        friend GT_FUNCTION positional operator+(positional const &lhs, positional const &rhs) {
            return {lhs.i + rhs.i, lhs.j + rhs.j, lhs.k + rhs.k};
        }

        struct stride_i {};
        struct stride_j {};
        struct stride_k {};

        friend GT_FUNCTION void sid_shift(positional &p, stride_i, int_t offset) { p.i += offset; }
        friend GT_FUNCTION void sid_shift(positional &p, stride_j, int_t offset) { p.j += offset; }
        friend GT_FUNCTION void sid_shift(positional &p, stride_k, int_t offset) { p.k += offset; }

        friend positional sid_get_origin(positional const &obj) { return obj; }

        friend hymap::keys<dim::i, dim::j, dim::k>::values<stride_i, stride_j, stride_k> sid_get_strides(
            positional const &) {
            return {};
        }
    };

    positional sid_get_ptr_diff(positional const &);

    GT_STATIC_ASSERT(is_sid<positional>::value, GT_INTERNAL_ERROR);
} // namespace gridtools
