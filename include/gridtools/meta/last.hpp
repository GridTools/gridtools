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

#include "at.hpp"
#include "defs.hpp"
#include "length.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        GT_META_LAZY_NAMESPACE {
            template <class List>
            GT_META_DEFINE_ALIAS(last, at_c, (List, length<List>::value - 1));
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <class List>
        using last = typename lazy::at_c<List, length<List>::value - 1>::type;
#endif
    } // namespace meta
} // namespace gridtools
