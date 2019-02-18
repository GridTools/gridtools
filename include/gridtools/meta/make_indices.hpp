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

#include <cstddef>

#include "defs.hpp"
#include "iseq_to_list.hpp"
#include "length.hpp"
#include "list.hpp"
#include "macros.hpp"
#include "utility.hpp"

namespace gridtools {
    namespace meta {
        GT_META_LAZY_NAMESPACE {
            /**
             *  Make a list of integral constants of indices from 0 to N
             */
            template <std::size_t N, template <class...> class L = list>
            GT_META_DEFINE_ALIAS(make_indices_c, iseq_to_list, (make_index_sequence<N>, L));

            template <class N, template <class...> class L = list>
            GT_META_DEFINE_ALIAS(make_indices, iseq_to_list, (make_index_sequence<N::value>, L));

            /**
             *  Make a list of integral constants of indices from 0 to length< List >
             */
            template <class List, template <class...> class L = list>
            GT_META_DEFINE_ALIAS(make_indices_for, iseq_to_list, (make_index_sequence<length<List>::value>, L));
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <std::size_t N, template <class...> class L = list>
        using make_indices_c = typename lazy::iseq_to_list<make_index_sequence<N>, L>::type;
        template <class N, template <class...> class L = list>
        using make_indices = typename lazy::iseq_to_list<make_index_sequence<N::value>, L>::type;
        template <class List, template <class...> class L = list>
        using make_indices_for = typename lazy::iseq_to_list<make_index_sequence<length<List>::value>, L>::type;
#endif
    } // namespace meta
} // namespace gridtools
