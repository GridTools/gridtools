/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include "macros.hpp"
#include "type_traits.hpp"

namespace gridtools {
    namespace meta {
        /**
         *  returns predicate that is the opposite of Pred
         */
        template <template <class...> class Pred>
        struct not_ {
            template <class T>
            using apply = negation<Pred<T>>;
        };
    } // namespace meta
} // namespace gridtools
