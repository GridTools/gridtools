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

#include "list.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Convert an integer sequence to a list of corresponding integral constants.
         */
        GT_META_LAZY_NAMESPACE {
            template <class, template <class...> class = list>
            struct iseq_to_list;
            template <template <class T, T...> class ISec, class Int, Int... Is, template <class...> class L>
            struct iseq_to_list<ISec<Int, Is...>, L> {
                using type = L<std::integral_constant<Int, Is>...>;
            };
        }
        GT_META_DELEGATE_TO_LAZY(iseq_to_list, (class ISec, template <class...> class L = list), (ISec, L));
    } // namespace meta
} // namespace gridtools
