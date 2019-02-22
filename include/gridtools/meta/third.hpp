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

#include "macros.hpp"

namespace gridtools {
    namespace meta {
        GT_META_LAZY_NAMESPACE {
            template <class>
            struct third;
            template <template <class...> class L, class T, class U, class Q, class... Ts>
            struct third<L<T, U, Q, Ts...>> {
                using type = Q;
            };
        }
        GT_META_DELEGATE_TO_LAZY(third, (class List), (List));
    } // namespace meta
} // namespace gridtools
