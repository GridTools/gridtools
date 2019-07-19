/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <utility>

#include "../common/hymap.hpp"
#include "../common/tuple.hpp"
#include "../common/tuple_util.hpp"
#include "dim.hpp"
#include "positional.hpp"
#include "sid/sid_shift_origin.hpp"

namespace gridtools {
    namespace backend_impl_ {
        template <class Grid, class DataStores>
        auto shift_origin(Grid const &grid, DataStores data_stores) {
            return tuple_util::transform([offsets = tuple_util::make<hymap::keys<dim::i, dim::j, dim::k>::values>(
                                              grid.i_low_bound(), grid.j_low_bound(), grid.k_min())](
                                             auto &src) { return sid::shift_sid_origin(std::ref(src), offsets); },
                std::move(data_stores));
        }

        template <class Grid>
        auto make_positionals(Grid const &grid, meta::list<dim::i, dim::j, dim::k>) {
            using positionals_t = tuple<positional<dim::i>, positional<dim::j>, positional<dim::k>>;
            return hymap::convert_to<hymap::keys, positionals_t>(
                positionals_t{grid.i_low_bound(), grid.j_low_bound(), grid.k_min()});
        }

        template <class Grid>
        tuple<> make_positionals(Grid &&, meta::list<>) {
            return {};
        }

        template <class Backend, class NeedPositionals, class Msses>
        struct backend_entry_point_f {
            template <class Grid, class DataStores>
            void operator()(Grid const &grid, DataStores data_stores) const {
                using positionals_t = meta::if_<NeedPositionals, meta::list<dim::i, dim::j, dim::k>, meta::list<>>;
                gridtools_backend_entry_point(Backend(),
                    Msses(),
                    grid,
                    shift_origin(grid, std::move(data_stores)),
                    make_positionals(grid, positionals_t()));
            }
        };
    } // namespace backend_impl_
    using backend_impl_::backend_entry_point_f;
} // namespace gridtools
