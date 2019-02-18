/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
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
        GT_META_LAZY_NAMESPACE {
            template <template <class...> class, class...>
            struct lfold;
            template <template <class...> class, class...>
            struct rfold;
        }
        GT_META_DELEGATE_TO_LAZY(lfold, (template <class...> class F, class... Args), (F, Args...));
        GT_META_DELEGATE_TO_LAZY(rfold, (template <class...> class F, class... Args), (F, Args...));

        GT_META_LAZY_NAMESPACE {
            template <template <class...> class F>
            struct lfold<F> {
                using type = curry_fun<meta::lfold, F>;
            };
            template <template <class...> class F, class S, template <class...> class L>
            struct lfold<F, S, L<>> {
                using type = S;
            };
            template <template <class...> class F, class S, template <class...> class L, class T>
            struct lfold<F, S, L<T>> {
                using type = GT_META_CALL(F, (S, T));
            };
            template <template <class...> class F, class S, template <class...> class L, class T1, class T2>
            struct lfold<F, S, L<T1, T2>> {
                using type = GT_META_CALL(F, (GT_META_CALL(F, (S, T1)), T2));
            };
            template <template <class...> class F, class S, template <class...> class L, class T1, class T2, class T3>
            struct lfold<F, S, L<T1, T2, T3>> {
                using type = GT_META_CALL(F, (GT_META_CALL(F, (GT_META_CALL(F, (S, T1)), T2)), T3));
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
                using type = typename lfold<F,
                    GT_META_CALL(F, (GT_META_CALL(F, (GT_META_CALL(F, (GT_META_CALL(F, (S, T1)), T2)), T3)), T4)),
                    L<Ts...>>::type;
            };

            template <template <class...> class F>
            struct rfold<F> {
                using type = curry_fun<meta::rfold, F>;
            };
            template <template <class...> class F, class S, template <class...> class L>
            struct rfold<F, S, L<>> {
                using type = S;
            };
            template <template <class...> class F, class S, template <class...> class L, class T>
            struct rfold<F, S, L<T>> {
                using type = GT_META_CALL(F, (T, S));
            };
            template <template <class...> class F, class S, template <class...> class L, class T1, class T2>
            struct rfold<F, S, L<T1, T2>> {
                using type = GT_META_CALL(F, (T1, GT_META_CALL(F, (T2, S))));
            };
            template <template <class...> class F, class S, template <class...> class L, class T1, class T2, class T3>
            struct rfold<F, S, L<T1, T2, T3>> {
                using type = GT_META_CALL(F, (T1, GT_META_CALL(F, (T2, GT_META_CALL(F, (T3, S))))));
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
                using type = GT_META_CALL(F,
                    (T1,
                        GT_META_CALL(F,
                            (T2, GT_META_CALL(F, (T3, GT_META_CALL(F, (T4, typename rfold<F, S, L<Ts...>>::type))))))));
            };
        }
    } // namespace meta
} // namespace gridtools
