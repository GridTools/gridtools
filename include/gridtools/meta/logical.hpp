/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cstddef>
#include <type_traits>

#include "list.hpp"
#include "macros.hpp"
#include "rename.hpp"
#include "repeat.hpp"
#include "transform.hpp"
#include "type_traits.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  C++17 drop-offs
         *
         *  Note on conjunction_fast and disjunction_fast are like std counter parts but:
         *    - short-circuiting is not implemented as required by C++17 standard
         *    - amortized complexity is O(1) because of it [in terms of the number of template instantiations].
         */
        template <class... Ts>
        GT_META_DEFINE_ALIAS(conjunction_fast,
            std::is_same,
            (list<std::integral_constant<bool, Ts::value>...>,
                GT_META_CALL(repeat_c, (GT_SIZEOF_3_DOTS(Ts), std::true_type))));

        template <class... Ts>
        GT_META_DEFINE_ALIAS(disjunction_fast, negation, conjunction_fast<negation<Ts>...>);

        /**
         *   all elements in lists are true
         */
        template <class List>
        struct all : lazy::rename<conjunction_fast, List>::type {};

        /**
         *   some elements in lists are true
         */
        template <class List>
        struct any : lazy::rename<disjunction_fast, List>::type {};

        /**
         *  All elements satisfy predicate
         */
        template <template <class...> class Pred, class List>
        GT_META_DEFINE_ALIAS(all_of, all, (GT_META_CALL(transform, (Pred, List))));

        /**
         *  Some element satisfy predicate
         */
        template <template <class...> class Pred, class List>
        GT_META_DEFINE_ALIAS(any_of, any, (GT_META_CALL(transform, (Pred, List))));

        template <class List>
        struct all_are_same;

        template <template <class...> class L>
        struct all_are_same<L<>> : std::true_type {};

        template <template <class...> class L, class T, class... Ts>
        struct all_are_same<L<T, Ts...>>
            : std::is_same<list<Ts...>, GT_META_CALL(repeat_c, (GT_SIZEOF_3_DOTS(Ts), T))> {};

    } // namespace meta
} // namespace gridtools
