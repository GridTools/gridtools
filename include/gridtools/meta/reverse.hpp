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

#include <cstddef>

#include "macros.hpp"
#include "push_back.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   reverse algorithm.
         *   Complexity is O(N)
         *   Making specializations for the first M allows to divide complexity by M.
         *   At a moment M = 4 (in boost::mp11 implementation it is 10).
         *   For the optimizers: fill free to add more specializations if needed.
         */
        GT_META_LAZY_NAMESPACE {
            template <class>
            struct reverse;

            template <template <class...> class L>
            struct reverse<L<>> {
                using type = L<>;
            };
            template <template <class...> class L, class T>
            struct reverse<L<T>> {
                using type = L<T>;
            };
            template <template <class...> class L, class T0, class T1>
            struct reverse<L<T0, T1>> {
                using type = L<T1, T0>;
            };
            template <template <class...> class L, class T0, class T1, class T2>
            struct reverse<L<T0, T1, T2>> {
                using type = L<T2, T1, T0>;
            };
            template <template <class...> class L, class T0, class T1, class T2, class T3>
            struct reverse<L<T0, T1, T2, T3>> {
                using type = L<T3, T2, T1, T0>;
            };
            template <template <class...> class L, class T0, class T1, class T2, class T3, class T4>
            struct reverse<L<T0, T1, T2, T3, T4>> {
                using type = L<T4, T3, T2, T1, T0>;
            };
            template <template <class...> class L, class T0, class T1, class T2, class T3, class T4, class... Ts>
            struct reverse<L<T0, T1, T2, T3, T4, Ts...>>
                : push_back<typename reverse<L<Ts...>>::type, T4, T3, T2, T1, T0> {};
        }
        GT_META_DELEGATE_TO_LAZY(reverse, class List, List);
    } // namespace meta
} // namespace gridtools
