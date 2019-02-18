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

#include "macros.hpp"
#include "utility.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Convert a list of integral constants to an integer sequence.
         */
        GT_META_LAZY_NAMESPACE {
            template <class>
            struct list_to_iseq;
            template <template <class...> class L, template <class T, T> class Const, class Int, Int... Is>
            struct list_to_iseq<L<Const<Int, Is>...>> {
                using type = integer_sequence<Int, Is...>;
            };
            template <template <class...> class L>
            struct list_to_iseq<L<>> {
                using type = index_sequence<>;
            };
        }
        GT_META_DELEGATE_TO_LAZY(list_to_iseq, class List, List);
    } // namespace meta
} // namespace gridtools
