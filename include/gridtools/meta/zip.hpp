/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "macros.hpp"
#include "transform.hpp"

namespace gridtools {
    /**
     *  Zip lists
     */
    namespace meta {
#if defined(__CUDACC_VER_MAJOR__) && \
    ((__CUDACC_VER_MAJOR__ == 9) || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ == 0))
        GT_META_LAZY_NAMESPACE {
            template <class... Lists>
            struct zip : transpose<list<Lists...>> {};
        }
        GT_META_DELEGATE_TO_LAZY(zip, class... Lists, Lists...);
#else
        GT_META_LAZY_NAMESPACE {
            template <class... Lists>
            GT_META_DEFINE_ALIAS(zip, transpose, list<Lists...>);
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <class... Lists>
        using zip = typename lazy::transpose<list<Lists...>>::type;
#endif
#endif
    } // namespace meta
} // namespace gridtools
