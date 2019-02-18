/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "curry.hpp"
#include "defer.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Extracts "producing template" from the list.
         *
         *  I.e ctor<some_instantiation_of_std_tuple>::apply is an alias of std::tuple.
         */
        GT_META_LAZY_NAMESPACE {
            template <class>
            struct ctor;
            template <template <class...> class L, class... Ts>
            struct ctor<L<Ts...>> : defer<L> {};
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <class>
        struct ctor;
        template <template <class...> class L, class... Ts>
        struct ctor<L<Ts...>> : curry<L> {};
#endif
    } // namespace meta
} // namespace gridtools
