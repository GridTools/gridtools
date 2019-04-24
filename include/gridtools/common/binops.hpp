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

#include "defs.hpp"
#include "generic_metafunctions/utility.hpp"
#include "host_device.hpp"

namespace gridtools {
    namespace binop {
        struct sum {
            template <class Lhs, class Rhs>
            GT_FUNCTION auto operator()(Lhs &&lhs, Rhs &&rhs) const
                GT_AUTO_RETURN(const_expr::forward<Lhs>(lhs) + const_expr::forward<Rhs>(rhs));
        };

        struct prod {
            template <class Lhs, class Rhs>
            GT_FUNCTION auto operator()(Lhs &&lhs, Rhs &&rhs) const
                GT_AUTO_RETURN(const_expr::forward<Lhs>(lhs) * const_expr::forward<Rhs>(rhs));
        };
    } // namespace binop
} // namespace gridtools
