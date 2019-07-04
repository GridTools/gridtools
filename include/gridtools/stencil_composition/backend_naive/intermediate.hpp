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
#include <tuple>
#include <utility>

#include "../../common/hymap.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../arg.hpp"
#include "../compute_extents_metafunctions.hpp"
#include "../dim.hpp"
#include "../esf_metafunctions.hpp"
#include "../extract_placeholders.hpp"
#include "../interval.hpp"
#include "../level.hpp"
#include "../loop_interval.hpp"
#include "../make_loop_intervals.hpp"
#include "../positional.hpp"
#include "../sid/allocator.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../sid/contiguous.hpp"
#include "../sid/loop.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "../stages_maker.hpp"

namespace gridtools {
    namespace naive {
        template <class Mss>
        using get_esfs = typename Mss::esf_sequence_t;

        template <class Plh, class Extent, class Grid>
        auto tmp_sizes(Grid const &grid) {
#ifndef GT_ICOSAHEDRAL_GRIDS
            return tuple_util::make<hymap::keys<dim::k, dim::j, dim::i>::values>(grid.k_max() + 1,
                grid.j_size() + typename Extent::jplus{} - typename Extent::jminus{},
                grid.i_size() + typename Extent::iplus{} - typename Extent::iminus{});
#else
            return tuple_util::make<hymap::keys<dim::c, dim::k, dim::j, dim::i>::values>(
                integral_constant<int, Plh::location_t::n_colors::value>{},
                grid.k_max() + 1,
                grid.j_size() + typename Extent::jplus{} - typename Extent::jminus{},
                grid.i_size() + typename Extent::iplus{} - typename Extent::iminus{});
#endif
        }

        template <class Grid, class DataStoreMap>
        auto shift_origin(Grid const &grid, DataStoreMap data_stores) {
            return tuple_util::transform(
                [offsets = tuple_util::make<hymap::keys<dim::i, dim::j>::values>(grid.i_low_bound(),
                     grid.j_low_bound())](auto &src) { return sid::shift_sid_origin(std::ref(src), offsets); },
                std::move(data_stores));
        }

        template <class Msses, class Grid, class Alloc>
        auto make_temporaries(Grid const &grid, Alloc &alloc) {
            using plhs_t = meta::filter<is_tmp_arg, extract_placeholders_from_msses<Msses>>;
            return tuple_util::transform(
                [&](auto plh) {
                    using plh_t = decltype(plh);
                    using extent_map_t = get_extent_map<meta::flatten<meta::transform<get_esfs, Msses>>>;
                    using extent_t = lookup_extent_map<extent_map_t, plh_t>;
                    using data_t = typename plh_t::data_store_t::data_t;
                    using stride_kind = meta::list<extent_t, typename plh_t::location_t::n_colors>;
                    using offsets_t =
                        hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, -extent_t::iminus::value>,
                            integral_constant<int_t, -extent_t::jminus::value>>;
                    return sid::shift_sid_origin(
                        sid::make_contiguous<data_t, ptrdiff_t, stride_kind>(alloc, tmp_sizes<plh_t, extent_t>(grid)),
                        offsets_t());
                },
                hymap::from_keys_values<plhs_t, plhs_t>());
        }

        template <class Grid>
        auto make_positionals(Grid const &grid) {
            using positionals_t = std::tuple<positional<dim::i>, positional<dim::j>, positional<dim::k>>;
            return hymap::convert_to<hymap::keys, positionals_t>(
                positionals_t{grid.i_low_bound(), grid.j_low_bound(), 0});
        }

        template <class IsStateful, class Grid, class Msses>
        auto make_intermediate(backend, IsStateful, Grid const &grid, Msses) {
            return [grid](auto data_stores) {
                auto alloc = sid::make_allocator(&std::make_unique<char[]>);

                auto composite = hymap::concat(sid::composite::keys<>::values<>(),
                    shift_origin(grid, std::move(data_stores)),
                    make_temporaries<Msses>(grid, alloc),
                    make_positionals(grid));

                auto origin = sid::get_origin(composite);
                auto strides = sid::get_strides(composite);

                for_each<Msses>([&](auto mss) {
                    using mss_t = decltype(mss);
                    using extent_map_t = get_extent_map<get_esfs<mss_t>>;
                    using execution_engine_t = typename mss_t::execution_engine_t;
                    using default_interval_t = interval<typename Grid::axis_type::FromLevel,
                        index_to_level<typename level_to_index<typename Grid::axis_type::ToLevel>::prior>>;
                    using loop_intervals_t = order_loop_intervals<execution_engine_t,
                        make_loop_intervals<stages_maker<mss_t, extent_map_t>::template apply, default_interval_t>>;

                    for_each<loop_intervals_t>([&](auto loop_interval) {
                        using loop_interval_t = decltype(loop_interval);
                        using from_t = meta::first<loop_interval_t>;
                        using to_t = meta::second<loop_interval_t>;
                        using stages_t = meta::third<loop_interval_t>;
                        for_each<stages_t>([&grid, &origin, &strides](auto stage) {
                            using extent_t = typename decltype(stage)::extent_t;

                            auto i_loop = sid::make_loop<dim::i>(
                                grid.i_size() + extent_t::iplus::value - extent_t::iminus::value);
                            auto j_loop = sid::make_loop<dim::j>(
                                grid.j_size() + extent_t::jplus::value - extent_t::jminus::value);
                            auto k_loop =
                                sid::make_loop<dim::k>(grid.count(from_t{}, to_t{}), execute::step<execution_engine_t>);

                            auto ptr = origin();
                            sid::shift(ptr, sid::get_stride<dim::i>(strides), typename extent_t::iminus{});
                            sid::shift(ptr, sid::get_stride<dim::j>(strides), typename extent_t::jminus{});
                            sid::shift(ptr, sid::get_stride<dim::k>(strides), grid.template value_at<from_t>());

                            i_loop(j_loop(k_loop(stage)))(ptr, strides);
                        });
                    });
                });
            };
        }
    } // namespace naive
} // namespace gridtools
