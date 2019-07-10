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
            GT_FUNCTION constexpr auto operator()(Lhs &&lhs, Rhs &&rhs) const {
                return std::forward<Lhs>(lhs) + std::forward<Rhs>(rhs);
            }
        };

        struct prod {
            template <class Lhs, class Rhs>
            GT_FUNCTION constexpr auto operator()(Lhs &&lhs, Rhs &&rhs) const {
                return std::forward<Lhs>(lhs) * std::forward<Rhs>(rhs);
            }
        };
    } // namespace binop
} // namespace gridtools
