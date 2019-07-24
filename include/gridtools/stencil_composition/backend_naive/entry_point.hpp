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

#include <memory>
#include <utility>

#include "../../common/defs.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../dim.hpp"
#include "../sid/allocator.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../sid/contiguous.hpp"
#include "../sid/loop.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "../stage_matrix.hpp"

namespace gridtools {
    namespace naive {
        template <class PlhMap, class Grid, class Alloc>
        auto make_temporaries(Grid const &grid, Alloc &alloc) {
            using plhs_t = meta::transform<stage_matrix::get_plh, PlhMap>;

            return tuple_util::transform(
                [&](auto info) {
                    using info_t = decltype(info);
                    using extent_t = typename info_t::extent_t;
                    using data_t = typename info_t::data_t;
                    using stride_kind = meta::list<extent_t, typename info_t::num_colors_t>;
                    using offsets_t =
                        hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, -extent_t::iminus::value>,
                            integral_constant<int_t, -extent_t::jminus::value>>;
                    auto sizes = tuple_util::make<hymap::keys<dim::c, dim::k, dim::j, dim::i>::values>(
                        typename info_t::num_colors_t(),
                        grid.k_size(),
                        grid.j_size(extent_t()),
                        grid.i_size(extent_t()));

                    return sid::shift_sid_origin(
                        sid::make_contiguous<data_t, ptrdiff_t, stride_kind>(alloc, sizes), offsets_t());
                },
                hymap::from_keys_values<plhs_t, PlhMap>());
        }

        template <class Spec, class Grid, class DataStores>
        void gridtools_backend_entry_point(backend, Spec, Grid const &grid, DataStores data_stores) {
            using stages_t = stage_matrix::make_split_view<Spec>;

            auto alloc = sid::make_allocator(&std::make_unique<char[]>);

            auto composite = hymap::concat(sid::composite::keys<>::values<>(),
                std::move(data_stores),
                make_temporaries<typename stages_t::tmp_plh_map_t>(grid, alloc));

            auto origin = sid::get_origin(composite);
            auto strides = sid::get_strides(composite);

            for_each<stages_t>([&](auto stage) {
                for_each<typename decltype(stage)::cells_t>([&](auto cell) {
                    using cell_t = decltype(cell);
                    using extent_t = typename cell_t::extent_t;
                    using execution_t = typename cell_t::execution_t;
                    using interval_t = typename cell_t::interval_t;
                    auto i_loop = sid::make_loop<dim::i>(grid.i_size(extent_t()));
                    auto j_loop = sid::make_loop<dim::j>(grid.j_size(extent_t()));
                    auto k_loop = sid::make_loop<dim::k>(grid.k_size(interval_t()), execute::step<execution_t>);

                    auto ptr = origin();
                    sid::shift(ptr, sid::get_stride<dim::i>(strides), typename extent_t::iminus());
                    sid::shift(ptr, sid::get_stride<dim::j>(strides), typename extent_t::jminus());
                    sid::shift(ptr, sid::get_stride<dim::k>(strides), grid.k_start(interval_t(), execution_t()));

                    i_loop(j_loop(k_loop([](auto &ptr, auto const &strides) {
                        for_each<typename cell_t::funs_t>([&](auto fun) { fun(ptr, strides); });
                    })))(ptr, strides);
                });
            });
        }
    } // namespace naive
} // namespace gridtools
