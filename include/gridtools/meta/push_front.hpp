/*
 * GridTools Libraries
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
            template <class...>
            struct push_front;
            template <template <class...> class L, class... Us, class... Ts>
            struct push_front<L<Us...>, Ts...> {
                using type = L<Ts..., Us...>;
            };
        }
        GT_META_DELEGATE_TO_LAZY(push_front, class... Args, Args...);
    } // namespace meta
} // namespace gridtools
