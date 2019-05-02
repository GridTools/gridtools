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
#include <type_traits>

#include "concat.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        GT_META_LAZY_NAMESPACE {
            template <size_t N, class List, class E = void>
            struct take_c;

            template <template <class...> class L, class... Ts>
            struct take_c<0, L<Ts...>> {
                using type = L<>;
            };

            template <template <class...> class L, class T0, class... Ts>
            struct take_c<1, L<T0, Ts...>> {
                using type = L<T0>;
            };

            template <template <class...> class L, class T0, class T1, class... Ts>
            struct take_c<2, L<T0, T1, Ts...>> {
                using type = L<T0, T1>;
            };

            template <template <class...> class L, class T0, class T1, class T2, class... Ts>
            struct take_c<3, L<T0, T1, T2, Ts...>> {
                using type = L<T0, T1, T2>;
            };

            template <template <class...> class L, class T0, class T1, class T2, class T3, class... Ts>
            struct take_c<4, L<T0, T1, T2, T3, Ts...>> {
                using type = L<T0, T1, T2, T3>;
            };

            template <size_t N,
                template <class...> class L,
                class T0,
                class T1,
                class T2,
                class T3,
                class T4,
                class... Ts>
            struct take_c<N, L<T0, T1, T2, T3, T4, Ts...>, typename std::enable_if<N >= 5>::type>
                : concat<L<T0, T1, T2, T3, T4>, typename take_c<N - 5, L<Ts...>>::type> {};

            template <class N, class List>
            GT_META_DEFINE_ALIAS(take, take_c, (N::value, List));
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <size_t N, class List>
        using take_c = typename lazy::take_c<N, List>::type;
        template <class N, class List>
        using take = typename lazy::take_c<N::value, List>::type;
#endif
    } // namespace meta
} // namespace gridtools
