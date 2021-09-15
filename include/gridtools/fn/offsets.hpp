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

#include <array>
#include <tuple>
#include <type_traits>

#include "../common/tuple_util.hpp"
#include "../meta.hpp"

namespace gridtools::fn {
    namespace offsets_impl_ {

        struct undefined {};

        undefined fn_offsets(...);

        inline constexpr auto offsets = []<class It>(It const &it) -> decltype(auto) {
            if constexpr (std::is_same_v<undefined, decltype(fn_offsets(it))>)
                return std::array<int, tuple_util::size<It>::value>{};
            else
                return fn_offsets(it);
        };
    } // namespace offsets_impl_
    using offsets_impl_::offsets;
} // namespace gridtools::fn
