/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "../stencil/positional.hpp"
#include "./backend/common.hpp"

namespace gridtools::fn {

    template <class T, class Allocator, class Sizes>
    auto allocate_global_tmp(Allocator &alloc, Sizes const &sizes) {
        return allocate_global_tmp(alloc, sizes, backend::data_type<T>());
    }

    template <int I, class Tuple>
    GT_FUNCTION auto &&tuple_get(integral_constant<int, I>, Tuple &&t) {
        return tuple_util::host_device::get<I>(std::forward<Tuple>(t));
    }

    template <class... Args>
    GT_FUNCTION tuple<std::decay_t<Args>...> make_tuple(Args &&...args) {
        return {std::forward<Args>(args)...};
    }

    template <class D>
    constexpr auto index(D) {
        return gridtools::stencil::positional<D>();
    }

} // namespace gridtools::fn
