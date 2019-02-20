/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
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
            struct first;
            template <template <class...> class L, class T, class... Ts>
            struct first<L<T, Ts...>> {
                using type = T;
            };
        }
        GT_META_DELEGATE_TO_LAZY(first, (class List), (List));
    } // namespace meta
} // namespace gridtools
