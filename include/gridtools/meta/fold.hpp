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

#include "curry_fun.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   Classic folds.
         *
         *   Complexity is O(N).
         *
         *   WARNING: Please use as a last resort. Consider `transform` ( which complexity is O(1) ) or `combine`
         *   (which has the same complexity but O(log(N)) template depth).
         */
        namespace lazy {
            template <template <class...> class, class...>
            struct lfold;
            template <template <class...> class, class...>
            struct rfold;
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(lfold, (template <class...> class F, class... Args), (F, Args...));
        GT_META_DELEGATE_TO_LAZY(rfold, (template <class...> class F, class... Args), (F, Args...));

        namespace lazy {
            template <template <class...> class F>
            struct lfold<F> {
                using type = curry_fun<meta::lfold, F>;
            };
            template <template <class...> class F>
            struct rfold<F> {
                using type = curry_fun<meta::rfold, F>;
            };
#if __cplusplus < 201703
            template <template <class...> class F, class S, template <class...> class L>
            struct lfold<F, S, L<>> {
                using type = S;
            };
            template <template <class...> class F, class S, template <class...> class L, class T>
            struct lfold<F, S, L<T>> {
                using type = F<S, T>;
            };
            template <template <class...> class F, class S, template <class...> class L, class T1, class T2>
            struct lfold<F, S, L<T1, T2>> {
                using type = F<F<S, T1>, T2>;
            };
            template <template <class...> class F, class S, template <class...> class L, class T1, class T2, class T3>
            struct lfold<F, S, L<T1, T2, T3>> {
                using type = F<F<F<S, T1>, T2>, T3>;
            };
            template <template <class...> class F,
                class S,
                template <class...> class L,
                class T1,
                class T2,
                class T3,
                class T4,
                class... Ts>
            struct lfold<F, S, L<T1, T2, T3, T4, Ts...>> {
                using type = typename lfold<F, F<F<F<F<S, T1>, T2>, T3>, T4>, L<Ts...>>::type;
            };
            template <template <class...> class F, class S, template <class...> class L>
            struct rfold<F, S, L<>> {
                using type = S;
            };
            template <template <class...> class F, class S, template <class...> class L, class T>
            struct rfold<F, S, L<T>> {
                using type = F<T, S>;
            };
            template <template <class...> class F, class S, template <class...> class L, class T1, class T2>
            struct rfold<F, S, L<T1, T2>> {
                using type = F<T1, F<T2, S>>;
            };
            template <template <class...> class F, class S, template <class...> class L, class T1, class T2, class T3>
            struct rfold<F, S, L<T1, T2, T3>> {
                using type = F<T1, F<T2, F<T3, S>>>;
            };
            template <template <class...> class F,
                class S,
                template <class...> class L,
                class T1,
                class T2,
                class T3,
                class T4,
                class... Ts>
            struct rfold<F, S, L<T1, T2, T3, T4, Ts...>> {
                using type = F<T1, F<T2, F<T3, F<T4, typename rfold<F, S, L<Ts...>>::type>>>>;
            };
#else
            namespace fold_impl_ {
                template <class>
                struct id;
                template <class>
                struct state;
                template <template <class...> class, class>
                struct folder;
                template <template <class...> class F, class S>
                struct state<folder<F, S> &&> {
                    using type = S;
                };
                template <template <class...> class F, class S, class T>
                folder<F, F<S, T>> &&operator+(folder<F, S> &&, id<T> *);
                template <template <class...> class F, class S, class T>
                folder<F, F<T, S>> &&operator+(id<T> *, folder<F, S> &&);
                template <class T>
                T &&make();
            } // namespace fold_impl_

            template <template <class...> class F, class S, template <class...> class L, class... Ts>
            struct lfold<F, S, L<Ts...>>
                : fold_impl_::state<decltype(
                      (fold_impl_::make<fold_impl_::folder<F, S>>() + ... + (fold_impl_::id<Ts> *)0))> {};

            template <template <class...> class F, class S, template <class...> class L, class... Ts>
            struct rfold<F, S, L<Ts...>>
                : fold_impl_::state<decltype(
                      ((fold_impl_::id<Ts> *)0 + ... + fold_impl_::make<fold_impl_::folder<F, S>>()))> {};
#endif
        } // namespace lazy
    }     // namespace meta
} // namespace gridtools
