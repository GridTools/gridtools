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

#include "../common/tuple_util.hpp"

namespace gridtools::fn {

    template <class T>
    concept Connectivity = requires(T const &conn) {
        fn_get_neighbour_offsets(conn, 0);
    }
    || requires(T const &conn) { conn[0]; };

    namespace connectivity_impl_ {
        struct undefined {};

        undefined fn_get_neighbour_offsets(...);
        undefined fn_get_output_range(...);

        template <Connectivity Conn, class Offset>
        constexpr decltype(auto) get_neighbour_offsets(Conn const &conn, Offset offset) {
            if constexpr (!std::is_same_v<decltype(fn_get_neighbour_offsets(conn, offset)), undefined>)
                return fn_get_neighbour_offsets(conn, offset);
            else
                return conn[offset];
        }

        template <Connectivity Conn, template <class...> class L, class... Is, class Range>
        Range default_get_output_range(Conn const &conn, L<Is...>, Range range) {
            int from = std::numeric_limits<int>::max();
            int to = 0;
            auto visit = [&](int v) {
                if (v == -1)
                    return;
                if (from == -1) {
                    from = v;
                    to = v + 1;
                    return;
                };
                from = std::min(from, v);
                to = std::max(to, v + 1);
            };
            for (int i = tuple_util::get<0>(range); i < tuple_util::get<1>(range); ++i) {
                auto &&offsets = get_neighbour_offsets(conn, i);
                (..., visit(tuple_util::get<Is::value>(offsets)));
            }
            return to ? Range{from, to} : Range{0, 0};
        }

        template <Connectivity Conn, class Neighbours, class Range>
        constexpr auto get_output_range(Conn const &conn, Neighbours, Range range) {
            if constexpr (!std::is_same_v<decltype(fn_get_output_range(conn, Neighbours(), range)), undefined>)
                return fn_get_output_range(conn, Neighbours(), range);
            else
                return default_get_output_range(conn, Neighbours(), range);
        }

        template <class T, class = void>
        struct neighbours_num : integral_constant<size_t, 0> {};

        template <class Conn>
        struct neighbours_num<Conn, std::enable_if_t<Connectivity<Conn>>>
            : tuple_util::size<std::decay_t<decltype(get_neighbour_offsets(std::declval<Conn const &>(), 0))>> {};

    } // namespace connectivity_impl_
    using connectivity_impl_::get_neighbour_offsets;
    using connectivity_impl_::get_output_range;
    using connectivity_impl_::neighbours_num;
} // namespace gridtools::fn
