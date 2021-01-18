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

#include "../../../../common/host_device.hpp"
#include "expr_base.hpp"

namespace gridtools {
    namespace stencil {
        namespace cartesian {
            namespace expressions {
                struct divide_f {
                    template <class Lhs, class Rhs>
                    GT_FUNCTION GT_CONSTEXPR auto operator()(Lhs const &lhs, Rhs const &rhs) const {
                        return lhs / rhs;
                    }
                };

                template <class Lhs, class Rhs>
                GT_FUNCTION GT_CONSTEXPR auto operator/(Lhs lhs, Rhs rhs)
                    -> decltype(make_expr(divide_f(), Lhs(), Rhs())) {
                    return make_expr(divide_f(), lhs, rhs);
                }
            } // namespace expressions
        }     // namespace cartesian
    }         // namespace stencil
} // namespace gridtools
