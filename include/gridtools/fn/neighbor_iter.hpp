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

#include <tuple>
#include <type_traits>

#include "builtins.hpp"
#include "connectivity.hpp"
#include "offsets.hpp"

namespace gridtools::fn {

    template <class Horizintal, class Offsets, class Impl>
    struct neighbors_iter {
        Offsets const &offsets;
        Impl impl;

        neighbors_iter(Horizintal, Offsets const &offsets, Impl impl) : offsets(offsets), impl(std::move(impl)) {}

        friend constexpr bool fn_builtin(builtins::can_deref, neighbors_iter const &it) { return false; }
        friend constexpr Offsets const &fn_offsets(neighbors_iter const &it) { return it.offsets; }
    };

    template <class Horizontal, class Impl>
    struct neighbor_iter {
        int offset;
        Impl impl;

        constexpr neighbor_iter(int offset, Impl impl) : offset(offset), impl(std::move(impl)) {}
        constexpr neighbor_iter(Horizontal, int offset, Impl impl) : offset(offset), impl(std::move(impl)) {}

        template <auto Dim, auto Val>
        friend constexpr neighbor_iter fn_builtin(builtins::shift<Dim, Val>, neighbor_iter const &it) {
            return {it.offset, shift<Dim, Val>(it.impl)};
        }

        template <Connectivity ConnT, ConnT Conn, auto Offset>
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

    template <class Horizontal, class Offsets, class Impl, auto I>
    constexpr auto fn_builtin(builtins::shift<I>, neighbors_iter<Horizontal, Offsets, Impl> const &it) {
        return neighbor_iter(Horizontal(), tuple_util::get<I>(offsets(it)), it.impl);
    }

    template <class Horizontal, class Impl, Connectivity ConnT, ConnT Conn>
    constexpr auto fn_builtin(builtins::shift<Conn>, neighbor_iter<Horizontal, Impl> const &it) {
        return neighbors_iter(Horizontal(), get_neighbour_offsets(Conn, it.offset), it.impl);
    }
} // namespace gridtools::fn
