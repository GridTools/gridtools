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

#include "defs.hpp"
#include "macros.hpp"
#include "pop_front.hpp"
#include "reverse.hpp"

namespace gridtools {
    namespace meta {
        GT_META_LAZY_NAMESPACE {
            template <class List>
            using pop_back = reverse<typename pop_front<typename reverse<List>::type>::type>;
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <class List>
        using pop_back =
            typename lazy::reverse<typename lazy::pop_front<typename lazy::reverse<List>::type>::type>::type;
#endif
    } // namespace meta
} // namespace gridtools
