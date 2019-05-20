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

#include <type_traits>

#include "macros.hpp"

namespace gridtools {
    namespace meta {
        GT_META_LAZY_NAMESPACE {
            /**
             *  Normalized std::conditional version, which is proper function in the terms of meta library.
             *
             *  Note: `std::conditional` should be named `if_c` according to `meta` name convention.
             */
            template <class Cond, class Lhs, class Rhs>
            using if_ = std::conditional<Cond::value, Lhs, Rhs>;

            template <bool Cond, class Lhs, class Rhs>
            using if_c = std::conditional<Cond, Lhs, Rhs>;
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <class Cond, class Lhs, class Rhs>
        using if_ = typename std::conditional<Cond::value, Lhs, Rhs>::type;

        template <bool Cond, class Lhs, class Rhs>
        using if_c = typename std::conditional<Cond, Lhs, Rhs>::type;
#endif
    } // namespace meta
} // namespace gridtools
