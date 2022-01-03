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

#include "host_device.hpp"

namespace gridtools {
    template <class>
    struct for_each_impl_;

    template <template <class...> class L, class... Ts>
    struct for_each_impl_<L<Ts...>> {
        template <class F>
        static GT_FORCE_INLINE constexpr void apply(F const &f) {
            (..., f(Ts()));
        }
    };

    template <class L, class F>
    GT_FORCE_INLINE constexpr void for_each(F const &f) {
        for_each_impl_<L>::apply(f);
    }
} // namespace gridtools
