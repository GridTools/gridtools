/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "defs.hpp"
#include "macros.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  Identity
         */
        GT_META_LAZY_NAMESPACE {
            template <class T>
            struct id {
                using type = T;
            };
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <class T>
        using id = T;
#endif
    } // namespace meta
} // namespace gridtools
