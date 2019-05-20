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

#include "defs.hpp"
#include "first.hpp"
#include "macros.hpp"
#include "make_indices.hpp"
#include "mp_find.hpp"
#include "second.hpp"
#include "zip.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   Take Nth element of the List
         */
        GT_META_LAZY_NAMESPACE {
            template <class List, std::size_t N>
            struct at_c;

            template <class List>
            struct at_c<List, 0> : first<List> {};

            template <class List>
            struct at_c<List, 1> : second<List> {};

            template <class List, std::size_t N>
            struct at_c : second<typename mp_find<typename zip<typename make_indices_for<List>::type, List>::type,
                              std::integral_constant<std::size_t, N>>::type> {};

            template <class List, class N>
            using at = at_c<List, N::value>;
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        // 'direct' versions of lazy functions
        template <class List, class N>
        using at = typename lazy::at_c<List, N::value>::type;
        template <class List, std::size_t N>
        using at_c = typename lazy::at_c<List, N>::type;
#endif
    } // namespace meta
} // namespace gridtools
