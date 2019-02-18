/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "defs.hpp"
#include "macros.hpp"
#include "pop_front.hpp"
#include "reverse.hpp"

namespace gridtools {
    namespace meta {
        GT_META_LAZY_NAMESPACE {
            template <class List>
            GT_META_DEFINE_ALIAS(pop_back, reverse, (typename pop_front<typename reverse<List>::type>::type));
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <class List>
        using pop_back =
            typename lazy::reverse<typename lazy::pop_front<typename lazy::reverse<List>::type>::type>::type;
#endif
    } // namespace meta
} // namespace gridtools
