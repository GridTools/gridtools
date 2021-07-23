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
    namespace shift_impl_ {

        struct undefined {};

        undefined fn_shift(...);

        constexpr auto shift() {
            return [](auto const &it) { return it; };
        };

        constexpr auto shift(auto offset) {
            return [offset](auto const &it) {
                if constexpr (!std::is_same_v<undefined, decltype(fn_shift(it, offset))>)
                    return fn_shift(it, offset);
                else if constexpr (std::is_convertible_v<decltype(offset), std::ptrdiff_t>)
                    return &it[offset];
            };
        };

        constexpr auto shift(auto offset0, auto offset1, auto... offsets) {
            return [offset0, offset1, offsets...](auto const &it) {
                if constexpr (!std::is_same_v<undefined, decltype(fn_shift(it, offset0, offset1, offsets...))>)
                    return fn_shift(it, offset0, offset1, offsets...);
                else if constexpr (!std::is_same_v<undefined, decltype(fn_shift(it, offset0, offset1))>)
                    return shift(offsets...)(fn_shift(it, offset0, offset1));
                else
                    return shift(offset1, offsets...)(fn_shift(it, offset0));
            };
        };
    } // namespace shift_impl_
    using shift_impl_::shift;
} // namespace gridtools::fn
