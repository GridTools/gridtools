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

    template <class T>
    concept Connectivity = requires(T const &conn) {
        fn_get_neighbour_offsets(conn, 0);
    }
    || requires(T const &conn) { conn[0]; };

    namespace connectivity_impl_ {
        struct undefined {};

        undefined fn_get_neighbour_offsets(...);

        template <Connectivity Conn, class Offset>
        decltype(auto) get_neighbour_offsets(Conn const &conn, Offset offset) {
            if constexpr (!std::is_same_v<decltype(fn_get_neighbour_offsets(conn, offset)), undefined>)
                return fn_get_neighbour_offsets(conn, offset);
            else
                return conn[offset];
        }
    } // namespace connectivity_impl_
    using connectivity_impl_::get_neighbour_offsets;
} // namespace gridtools::fn
