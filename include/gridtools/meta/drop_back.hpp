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

#include "defs.hpp"
#include "drop_front.hpp"
#include "macros.hpp"
#include "reverse.hpp"

namespace gridtools {
    namespace meta {
        namespace lazy {
            template <std::size_t N, class List>
            using drop_back_c = reverse<typename drop_front_c<N, typename reverse<List>::type>::type>;

            template <class N, class List>
            using drop_back = reverse<typename drop_front_c<N::value, typename reverse<List>::type>::type>;
        }
#if !GT_BROKEN_TEMPLATE_ALIASES
        template <std::size_t N, class List>
        using drop_back_c =
            typename lazy::reverse<typename lazy::drop_front_c<N, typename lazy::reverse<List>::type>::type>::type;

        template <class N, class List>
        using drop_back = typename lazy::reverse<
            typename lazy::drop_front_c<N::value, typename lazy::reverse<List>::type>::type>::type;
#endif
    } // namespace meta
} // namespace gridtools
