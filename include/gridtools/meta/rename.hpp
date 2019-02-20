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

#include "curry_fun.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   Instantiate F with the parameters taken from List.
         *
         *   Alternative interpretation: apply function F to the arguments taken form List.
         */
        GT_META_LAZY_NAMESPACE {
            template <template <class...> class, class...>
            struct rename;
        }
        GT_META_DELEGATE_TO_LAZY(rename, (template <class...> class F, class... Args), (F, Args...));
        GT_META_LAZY_NAMESPACE {
            template <template <class...> class To, template <class...> class From, class... Ts>
            struct rename<To, From<Ts...>> {
                using type = GT_META_CALL(To, Ts...);
            };
            template <template <class...> class To>
            struct rename<To> {
                using type = curry_fun<meta::rename, To>;
            };
        }
    } // namespace meta
} // namespace gridtools
