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
#include <tuple>
#include <type_traits>

#include "builtins.hpp"
#include "connectivity.hpp"
#include "offsets.hpp"

namespace gridtools::fn {

    template <Connectivity ConnT, ConnT Conn, class Nested>
    struct partial_iter {
        Nested nested;

        int horizontal_offset() const { return nested.horizontal_offset(); }

        friend constexpr bool fn_builtin(builtins::can_deref, partial_iter) { return false; }

        friend constexpr auto const &fn_offsets(partial_iter const &it) {
            return get_neighbour_offsets(Conn, it.horizontal_offset());
        }
    };

    template <class Horizontal, class Impl>
    struct neighbor_iter {
        int offset;
        Impl impl;

        constexpr neighbor_iter(int offset, Impl impl) : offset(offset), impl(std::move(impl)) {}
        constexpr neighbor_iter(Horizontal, int offset, Impl impl) : offset(offset), impl(std::move(impl)) {}

        int horizontal_offset() const { return offset; }

        template <auto Dim, auto Val>
        friend constexpr neighbor_iter fn_builtin(builtins::shift<Dim, Val>, neighbor_iter const &it) {
            return {it.offset, shift<Dim, Val>(it.impl)};
        }

        template <Connectivity ConnT, ConnT Conn, auto Offset>
        requires std::integral<decltype(Offset)>
        friend constexpr neighbor_iter fn_builtin(builtins::shift<Conn, Offset>, neighbor_iter const &it) {
            return {tuple_util::get<Offset>(get_neighbour_offsets(Conn, it.offset)), it.impl};
        }

        friend constexpr bool fn_builtin(builtins::can_deref, neighbor_iter const &it) {
            return it.offset != -1 && can_deref(it.impl);
        }
    };

    template <class Horizontal, class Impl>
    constexpr decltype(auto) fn_builtin(builtins::deref, neighbor_iter<Horizontal, Impl> const &it) {
        return it.impl.sid_access(Horizontal(), it.offset);
    }

    template <auto I, Connectivity ConnT, ConnT Conn, class Nested>
    requires std::integral<decltype(I)>
    constexpr auto fn_builtin(builtins::shift<I>, partial_iter<ConnT, Conn, Nested> const &it) {
        return shift<Conn, I>(it.nested);
    }

    template <Connectivity ConnT, ConnT Conn, auto I, Connectivity ConnT2, ConnT2 Conn2, class Nested>
    requires std::integral<decltype(I)>
    constexpr auto fn_builtin(builtins::shift<Conn, I>, partial_iter<ConnT2, Conn2, Nested> const &it) {
        auto nested = shift<Conn, I>(it.nested);
        return partial_iter<ConnT2, Conn2, decltype(nested)>(std::move(nested));
    }

    template <Connectivity ConnT, ConnT Conn, class Horizontal, class Impl>
    constexpr auto fn_builtin(builtins::shift<Conn>, neighbor_iter<Horizontal, Impl> const &it) {
        return partial_iter<ConnT, Conn, neighbor_iter<Horizontal, Impl>>{it};
    }

    template <Connectivity ConnT, ConnT Conn, Connectivity ConnT2, ConnT2 Conn2, class Nested>
    constexpr auto fn_builtin(builtins::shift<Conn>, partial_iter<ConnT2, Conn2, Nested> const &it) {
        return partial_iter<ConnT, Conn, partial_iter<ConnT2, Conn2, Nested>>{it};
    }
} // namespace gridtools::fn
