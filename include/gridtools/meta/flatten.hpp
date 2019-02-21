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

#include "combine.hpp"
#include "concat.hpp"
#include "defs.hpp"
#include "macros.hpp"

namespace gridtools {
    /**
     *  Flatten a list of lists.
     *
     *  Note: this function doesn't go recursive. It just concatenates the inner lists.
     */
    namespace meta {
        GT_META_LAZY_NAMESPACE {
            template <class Lists>
            GT_META_DEFINE_ALIAS(flatten, combine, (meta::concat, Lists));
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <class Lists>
        using flatten = typename lazy::combine<concat, Lists>::type;
#endif
    } // namespace meta
} // namespace gridtools
