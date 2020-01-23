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

#include "../../../../common/host_device.hpp"
#include "expr_base.hpp"

namespace gridtools {
    namespace cartesian {
        namespace expressions {
            struct minus_f {
                template <class Lhs, class Rhs>
                GT_FUNCTION GT_CONSTEXPR auto operator()(Lhs const &lhs, Rhs const &rhs) const {
                    return lhs - rhs;
                }
                template <class Arg>
                GT_FUNCTION GT_CONSTEXPR auto operator()(Arg const &arg) const {
                    return -arg;
                }
            };

            template <class Lhs, class Rhs>
            GT_FUNCTION GT_CONSTEXPR auto operator-(Lhs lhs, Rhs rhs) -> decltype(make_expr(minus_f(), Lhs(), Rhs())) {
                return make_expr(minus_f(), lhs, rhs);
            }

            template <class Arg>
            GT_FUNCTION GT_CONSTEXPR auto operator-(Arg arg) -> decltype(make_expr(minus_f(), Arg())) {
                return make_expr(minus_f(), arg);
            }
        } // namespace expressions
    }     // namespace cartesian
} // namespace gridtools
