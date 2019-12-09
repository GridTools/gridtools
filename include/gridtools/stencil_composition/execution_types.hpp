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

#include <type_traits>

#include "../common/defs.hpp"
#include "../common/integral_constant.hpp"
#include "../meta/type_traits.hpp"

namespace gridtools {
    namespace execute {
        struct parallel {};
        struct forward {};
        struct backward {};

        template <class T>
        using is_parallel = std::is_same<T, parallel>;

        template <class T>
        using is_forward = std::is_same<T, forward>;

        template <class T>
        using is_backward = std::is_same<T, backward>;

        template <class T>
        constexpr integral_constant<int_t, is_backward<T>::value ? -1 : 1> step = {};

        template <class T>
        using is_execution_engine = disjunction<is_parallel<T>, is_backward<T>, is_forward<T>>;
    } // namespace execute
} // namespace gridtools
