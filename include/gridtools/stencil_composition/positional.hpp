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
#include "sid/simple_ptr_holder.hpp"
#include "sid/synthetic.hpp"

namespace gridtools {
    namespace positional_impl_ {
        struct point {
            int_t i;
            int_t j;
            int_t k;

            point const &operator*() const { return *this; }

            friend GT_FUNCTION point operator+(point lhs, point rhs) {
                return {lhs.i + rhs.i, lhs.j + rhs.j, lhs.k + rhs.k};
            }
        };

        struct stride_i {};
        struct stride_j {};
        struct stride_k {};

        GT_FUNCTION void sid_shift(point &p, stride_i, int_t offset) { p.i += offset; }
        GT_FUNCTION void sid_shift(point &p, stride_j, int_t offset) { p.j += offset; }
        GT_FUNCTION void sid_shift(point &p, stride_k, int_t offset) { p.k += offset; }

        using sid::property;

        auto make_positional_sid(point p = {}) GT_AUTO_RETURN(
            (sid::synthetic()
                    .set<property::origin>(sid::make_simple_ptr_holder(p))
                    .set<property::strides>(hymap::keys<dim::i, dim::j, dim::k>::values<stride_i, stride_j, stride_k>{})
                    .set<property::ptr_diff, point>()));
    } // namespace positional_impl_

    using positional_impl_::make_positional_sid;

    using positional_sid_t = decltype(make_positional_sid());

    GT_STATIC_ASSERT(is_sid<positional_sid_t>::value, GT_INTERNAL_ERROR);
} // namespace gridtools
