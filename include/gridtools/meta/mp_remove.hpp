/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <type_traits>

#include "filter.hpp"
#include "first.hpp"
#include "macros.hpp"
#include "type_traits.hpp"

namespace gridtools {
    namespace meta {
        template <class Key>
        struct mp_remove_helper {
            template <class T>
            GT_META_DEFINE_ALIAS(apply, negation, (std::is_same<typename lazy::first<T>::type, Key>));
        };

        template <class Map, class Key>
        GT_META_DEFINE_ALIAS(mp_remove, filter, (mp_remove_helper<Key>::template apply, Map));
    } // namespace meta
} // namespace gridtools
