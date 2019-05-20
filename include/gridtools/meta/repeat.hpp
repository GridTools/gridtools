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

#include "defs.hpp"
#include "list.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Produce a list of N identical elements
         */
        GT_META_LAZY_NAMESPACE {
            template <class List, bool Rem>
            struct repeat_impl_expand;

            template <class... Ts>
            struct repeat_impl_expand<list<Ts...>, false> {
                using type = list<Ts..., Ts...>;
            };

            template <class T, class... Ts>
            struct repeat_impl_expand<list<T, Ts...>, true> {
                using type = list<T, T, T, Ts..., Ts...>;
            };

            template <std::size_t N, class T>
            struct repeat_c {
                using type = typename repeat_impl_expand<typename repeat_c<N / 2, T>::type, N % 2>::type;
            };

            template <class T>
            struct repeat_c<0, T> {
                using type = list<>;
            };

            template <class T>
            struct repeat_c<1, T> {
                using type = list<T>;
            };

            template <class N, class T>
            using repeat = repeat_c<N::value, T>;
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <std::size_t N, class T>
        using repeat_c = typename lazy::repeat_c<N, T>::type;
        template <class N, class T>
        using repeat = typename lazy::repeat_c<N::value, T>::type;
#endif
    } // namespace meta
} // namespace gridtools
