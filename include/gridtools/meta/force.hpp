/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

namespace gridtools {
    namespace meta {
        /**
         *  Remove laziness from a function
         */
        template <template <class...> class F>
        struct force {
            template <class... Args>
            using apply = typename F<Args...>::type;
        };
    } // namespace meta
} // namespace gridtools
