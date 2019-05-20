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

#include <type_traits>

#include "clear.hpp"
#include "fold.hpp"
#include "list.hpp"
#include "macros.hpp"
#include "mp_insert.hpp"

namespace gridtools {
    namespace meta {
        GT_META_LAZY_NAMESPACE {
            template <class State, class Item>
            struct mp_inverse_helper;

            template <class State, template <class...> class L, class Key, class... Vals>
            struct mp_inverse_helper<State, L<Key, Vals...>>
                : lfold<meta::mp_insert, State, meta::list<L<Vals, Key>...>> {};
        }
        GT_META_DELEGATE_TO_LAZY(mp_inverse_helper, (class State, class Item), (State, Item));

        template <class Src>
        using mp_inverse = lfold<mp_inverse_helper, clear<Src>, Src>;
    } // namespace meta
} // namespace gridtools
