/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <type_traits>

#include "curry.hpp"
#include "type_traits.hpp"

namespace gridtools {
    namespace meta {
        /**
         *   meta class concept check
         */
        template <class, class = void>
        struct is_meta_class : std::false_type {};
        template <class T>
        struct is_meta_class<T, void_t<curry<T::template apply>>> : std::true_type {};
    } // namespace meta
} // namespace gridtools
