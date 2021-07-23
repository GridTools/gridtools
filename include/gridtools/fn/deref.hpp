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

#include <type_traits>

namespace gridtools::fn {
    namespace deref_impl_ {

        struct undefined {};

        undefined fn_deref(...);

        undefined fn_can_deref(...);

        inline constexpr auto deref = [](auto const &it) {
            if constexpr (!std::is_same_v<undefined, decltype(fn_deref(it))>)
                return fn_deref(it);
            else
                return *it;
        };

        inline constexpr auto can_deref = [](auto const &it) -> bool {
            if constexpr (!std::is_same_v<undefined, decltype(fn_can_deref(it))>)
                return fn_can_deref(it);
            else if constexpr (std::is_constructible_v<bool, decltype(it)>)
                return bool(it);
            else
                return true;
        };

    } // namespace deref_impl_
    using deref_impl_::can_deref;
    using deref_impl_::deref;
} // namespace gridtools::fn
