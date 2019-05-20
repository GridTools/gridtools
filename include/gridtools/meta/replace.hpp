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

#include <boost/preprocessor.hpp>

#include "always.hpp"
#include "curry.hpp"
#include "defs.hpp"
#include "first.hpp"
#include "force.hpp"
#include "if.hpp"
#include "macros.hpp"
#include "make_indices.hpp"
#include "transform.hpp"

#if GT_BROKEN_TEMPLATE_ALIASES
#define GT_META_INTERNAL_LAZY_PARAM(fun) BOOST_PP_REMOVE_PARENS(fun)
#else
#define GT_META_INTERNAL_LAZY_PARAM(fun) ::gridtools::meta::force<BOOST_PP_REMOVE_PARENS(fun)>::template apply
#endif

namespace gridtools {
    namespace meta {
        template <template <class...> class Pred, template <class...> class F>
        struct selective_call_impl {
            template <class Arg>
            GT_META_DEFINE_ALIAS(apply, meta::if_, (Pred<Arg>, F<Arg>, Arg));
        };

        template <template <class...> class Pred, template <class...> class F, class List>
        GT_META_DEFINE_ALIAS(selective_transform, transform, (selective_call_impl<Pred, F>::template apply, List));

        /**
         *   replace all Old elements to New within List
         */
        template <class List, class Old, class New>
        GT_META_DEFINE_ALIAS(replace,
            selective_transform,
            (curry<std::is_same, Old>::template apply, always<New>::template apply, List));

        template <class Key>
        struct is_same_key_impl {
            template <class Elem>
            GT_META_DEFINE_ALIAS(apply, std::is_same, (Key, first<Elem>));
        };

        template <class... NewVals>
        struct replace_values_impl {
            template <class MapElem>
            struct apply;
            template <template <class...> class L, class Key, class... OldVals>
            struct apply<L<Key, OldVals...>> {
                using type = L<Key, NewVals...>;
            };
        };

        /**
         *  replace element in the map by key
         */
        template <class Map, class Key, class... NewVals>
        GT_META_DEFINE_ALIAS(mp_replace,
            selective_transform,
            (is_same_key_impl<Key>::template apply,
                GT_META_INTERNAL_LAZY_PARAM(replace_values_impl<NewVals...>::template apply),
                Map));

        template <std::size_t N, class New>
        struct replace_at_impl {
            template <class T, class I>
            struct apply {
                using type = T;
            };
            template <class T>
            struct apply<T, std::integral_constant<std::size_t, N>> {
                using type = New;
            };
        };

        /**
         *  replace element at given position
         */
        template <class List, std::size_t N, class New>
        GT_META_DEFINE_ALIAS(replace_at_c,
            transform,
            (GT_META_INTERNAL_LAZY_PARAM((replace_at_impl<N, New>::template apply)),
                List,
                typename make_indices_for<List>::type));

        template <class List, class N, class New>
        GT_META_DEFINE_ALIAS(replace_at, replace_at_c, (List, N::value, New));
    } // namespace meta
} // namespace gridtools

#undef GT_META_INTERNAL_LAZY_PARAM
