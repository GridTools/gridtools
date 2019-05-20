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

#include "first.hpp"
#include "macros.hpp"
#include "mp_find.hpp"
#include "mp_remove.hpp"
#include "push_back.hpp"

namespace gridtools {
    namespace meta {
        GT_META_LAZY_NAMESPACE {
            template <class Map, class Elem, class OldElem = typename mp_find<Map, typename first<Elem>::type>::type>
            struct mp_insert;

            template <class Map, template <class...> class L, class Key, class... Vals, class OldElem>
            struct mp_insert<Map, L<Key, Vals...>, OldElem> {
                using type = typename push_back<mp_remove<Map, Key>, typename push_back<OldElem, Vals...>::type>::type;
            };

            template <class Map, template <class...> class L, class Key, class... Vals>
            struct mp_insert<Map, L<Key, Vals...>, void> : push_back<Map, L<Key, Vals...>> {};
        }
        GT_META_DELEGATE_TO_LAZY(
            mp_insert, (class Map, class Elem, class OldElem = mp_find<Map, first<Elem>>), (Map, Elem, OldElem));
    } // namespace meta
} // namespace gridtools
