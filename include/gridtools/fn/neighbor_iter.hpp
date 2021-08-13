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

#include "connectivity.hpp"
#include "deref.hpp"
#include "offsets.hpp"
#include "shift.hpp"

namespace gridtools::fn {

    template <class Horizintal, class Offsets, class Impl>
    struct neighbors_iter {
        Offsets const &offsets;
        Impl impl;

        neighbors_iter(Horizintal, Offsets const &offsets, Impl impl) : offsets(offsets), impl(std::move(impl)) {}

        friend constexpr bool fn_can_deref(neighbors_iter const &it) { return false; }
        friend constexpr Offsets const &fn_offsets(neighbors_iter const &it) { return it.offsets; }
    };

    template <class Horizontal, class Impl>
    struct neighbor_iter {
        int offset;
        Impl impl;

        constexpr neighbor_iter(int offset, Impl impl) : offset(offset), impl(std::move(impl)) {}
        constexpr neighbor_iter(Horizontal, int offset, Impl impl) : offset(offset), impl(std::move(impl)) {}

        template <class Dim, class Offset>
        friend constexpr neighbor_iter fn_shift(neighbor_iter const &it, Dim const &d, Offset val) {
            return {it.offset, shift(d, val)(it.impl)};
        }

        template <Connectivity Conn, class Offset>
        friend constexpr neighbor_iter fn_shift(neighbor_iter const &it, Conn const &conn, Offset) {
            return {tuple_util::get<Offset::value>(get_neighbour_offsets(conn, it.offset)), it.impl};
        }

        friend constexpr bool fn_can_deref(neighbor_iter const &it) { return it.offset != -1 && can_deref(it.impl); }
    };

    template <class Horizontal, class Impl>
    constexpr decltype(auto) fn_deref(neighbor_iter<Horizontal, Impl> const &it) {
        return deref(shift(Horizontal(), it.offset)(it.impl));
    }

    template <class Horizontal, class Offsets, class Impl, size_t I>
    constexpr auto fn_shift(neighbors_iter<Horizontal, Offsets, Impl> const &it, fast_offset<I> value) {
        return neighbor_iter(Horizontal(), value.offset, it.impl);
    }

    template <class Horizontal, class Offsets, class Impl, class T, T I>
    constexpr auto fn_shift(neighbors_iter<Horizontal, Offsets, Impl> const &it, std::integral_constant<T, I>) {
        return neighbor_iter(Horizontal(), tuple_util::get<I>(offsets(it)), it.impl);
    }

    template <class Horizontal, class Impl, Connectivity Conn>
    constexpr auto fn_shift(neighbor_iter<Horizontal, Impl> const &it, Conn const &conn) {
        return neighbors_iter(Horizontal(), get_neighbour_offsets(conn, it.offset), it.impl);
    }
} // namespace gridtools::fn
