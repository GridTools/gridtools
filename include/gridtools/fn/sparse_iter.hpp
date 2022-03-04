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

#include <concepts>

#include "builtins.hpp"
#include "connectivity.hpp"
#include "neighbor_iter.hpp"
#include "offsets.hpp"

namespace gridtools::fn {

    template <class Nested>
    struct sparse_iter {
        Nested nested;
    };

    template <class Nested, auto Index>
    struct shifted_sparse_iter {
        Nested nested;
    };

    template <class Nested>
    constexpr bool fn_builtin(builtins::can_deref, sparse_iter<Nested>) {
        return false;
    }

    template <auto I, class Nested>
    requires std::integral<decltype(I)>
    constexpr auto fn_builtin(builtins::shift<I>, sparse_iter<Nested> const &it) {
        return shifted_sparse_iter<Nested, I>{it.nested};
    }

    template <Connectivity ConnT, ConnT Conn, auto I, class Nested>
    requires std::integral<decltype(I)>
    constexpr sparse_iter<Nested> fn_builtin(builtins::shift<Conn, I>, sparse_iter<Nested> const &it) {
        return {shift<Conn, I>(it.nested)};
    }

    template <Connectivity ConnT, ConnT Conn, class Nested>
    constexpr partial_iter<ConnT, Conn, sparse_iter<Nested>> fn_builtin(
        builtins::shift<Conn>, sparse_iter<Nested> const &it) {
        return {it};
    }

    template <class Nested, auto Index>
    constexpr bool fn_builtin(builtins::can_deref, shifted_sparse_iter<Nested, Index> const &it) {
        return can_deref(it.nested);
    }

    template <class Nested, auto Index>
    constexpr decltype(auto) fn_builtin(builtins::deref, shifted_sparse_iter<Nested, Index> const &it) {
        return tuple_util::get<Index>(deref(it.nested));
    }

    template <auto... Shifts, class Nested, auto Index>
    constexpr shifted_sparse_iter<Nested, Index> fn_builtin(
        builtins::shift<Shifts...>, shifted_sparse_iter<Nested, Index> const &it) {
        return {shift<Shifts...>(it.nested)};
    }

    template <class Arg>
    constexpr sparse_iter<Arg> fn_builtin(builtins::sparse, Arg const &arg) {
        return {arg};
    }
} // namespace gridtools::fn
