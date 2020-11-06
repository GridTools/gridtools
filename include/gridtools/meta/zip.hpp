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

#include "macros.hpp"
#include "transform.hpp"

namespace gridtools {
    /**
     *  Zip lists
     */
    namespace meta {
#if defined(__NVCC__) && defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ < 10 || __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ < 2)
        namespace lazy {
            template <class... Lists>
            struct zip : transpose<list<Lists...>> {};
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(zip, class... Lists, Lists...);
#else
        namespace lazy {
            template <class... Lists>
            using zip = transpose<list<Lists...>>;
        }
        template <class... Lists>
        using zip = typename lazy::transpose<list<Lists...>>::type;
#endif
    } // namespace meta
} // namespace gridtools
