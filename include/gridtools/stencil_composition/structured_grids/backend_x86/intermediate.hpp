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

#include <type_traits>

#include "../../../common/defs.hpp"
#include "../../../common/tuple_util.hpp"
#include "../../../meta.hpp"
#include "../../arg.hpp"
#include "../../compute_extents_metafunctions.hpp"
#include "../../esf_metafunctions.hpp"
#include "../../extent.hpp"
#include "../../extract_placeholders.hpp"
#include "../../grid.hpp"
#include "../../mss_components.hpp"
#include "../../mss_components_metafunctions.hpp"
#include "../../sid/allocator.hpp"
#include "../../sid/block.hpp"
#include "../../sid/contiguous.hpp"
#include "../../sid/sid_shift_origin.hpp"
#include "mss_loop_x86.hpp"

namespace gridtools {
    namespace x86 {
        template <class Mss>
        using get_esfs = typename Mss::esf_sequence_t;

        struct block_f {
            template <class T>
            auto operator()(T &&data_store) const {
                return sid::block(std::forward<T>(data_store),
                    hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, GT_DEFAULT_TILE_I>,
                        integral_constant<int_t, GT_DEFAULT_TILE_J>>());
            }
        };

        template <class Grid, class DataStoreMap>
        auto shift_origin_and_block(Grid const &grid, DataStoreMap data_stores) {
            return tuple_util::transform(
                [offsets = tuple_util::make<hymap::keys<dim::i, dim::j, dim::k>::values>(
                     grid.i_low_bound(), grid.j_low_bound(), grid.k_min()),
                    block = block_f()](auto &src) { return block(sid::shift_sid_origin(std::ref(src), offsets)); },
                std::move(data_stores));
        }

        template <class Msses, class Grid, class Allocator>
        auto make_temporaries(Grid const &grid, Allocator &allocator) {
            using plhs_t = meta::filter<is_tmp_arg, extract_placeholders_from_msses<Msses>>;
            using extent_map_t = get_extent_map<meta::flatten<meta::transform<get_esfs, Msses>>>;
            return tuple_util::transform(
                [&allocator, &grid](auto plh) {
                    using plh_t = decltype(plh);
                    using data_t = typename plh_t::data_store_t::data_t;
                    using extent_t = lookup_extent_map<extent_map_t, plh_t>;
                    return sid::shift_sid_origin(
                        sid::make_contiguous<data_t, int_t, extent_t>(allocator,
                            tuple_util::make<hymap::keys<dim::k, dim::j, dim::i, dim::thread>::values>(
                                grid.k_total_length(),
                                integral_constant<int_t,
                                    GT_DEFAULT_TILE_J + extent_t::jplus::value - extent_t::jminus::value>(),
                                integral_constant<int_t,
                                    GT_DEFAULT_TILE_I + extent_t::iplus::value - extent_t::iminus::value>(),
                                omp_get_max_threads())),
                        hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, -extent_t::iminus::value>,
                            integral_constant<int_t, -extent_t::jminus::value>>());
                },
                hymap::from_keys_values<plhs_t, plhs_t>());
        }

        template <class Grid>
        auto make_positionals(meta::list<dim::i, dim::j, dim::k>, Grid const &grid) {
            using positionals_t = tuple<positional<dim::i>, positional<dim::j>, positional<dim::k>>;
            return tuple_util::transform(block_f(),
                hymap::convert_to<hymap::keys, positionals_t>(
                    positionals_t{grid.i_low_bound(), grid.j_low_bound(), grid.k_min()}));
        }

        template <class Grid>
        tuple<> make_positionals(meta::list<>, Grid const &) {
            return {};
        }

        template <class DataStores>
        struct at_key_f {
            DataStores &m_src;
            template <class Plh>
            GT_FORCE_INLINE decltype(auto) operator()(Plh) const {
                return at_key<Plh>(m_src);
            }
        };

        template <class Plhs, class Src>
        auto filter_map(Src &src) {
            return tuple_util::transform(at_key_f<Src>{src}, hymap::from_keys_values<Plhs, Plhs>());
        }

        template <class Mss, class NeedPositionals, class Grid, class DataStores>
        auto make_composite(Grid grid, DataStores &data_stores) {
            using positionals_t = meta::if_<NeedPositionals, meta::list<dim::i, dim::j, dim::k>, meta::list<>>;

            using plhs_t = extract_placeholders_from_mss<Mss>;

            return hymap::concat(sid::composite::keys<>::values<>(),
                filter_map<plhs_t>(data_stores),
                make_positionals(positionals_t(), grid));
        }

        template <class Count, class Stages>
        struct loop_interval {
            Count m_count;
            GT_FORCE_INLINE Count count() const { return m_count; }
            template <class Ptr, class Strides>
            GT_FORCE_INLINE void operator()(Ptr const &GT_RESTRICT ptr, Strides const &GT_RESTRICT strides) const {
                for_each<Stages>([&](auto stage) { stage(ptr, strides); });
            }
        };

        template <class Grid>
        struct adapt_interval_f {
            Grid const &m_grid;
            template <class Interval>
            auto operator()(Interval) const {
                auto count = m_grid.count(meta::first<Interval>(), meta::second<Interval>());
                return loop_interval<decltype(count), meta::third<Interval>>{count};
            }
        };

        template <class MssComponents, class Grid>
        auto make_loop_intervals(Grid const &grid) {
            return tuple_util::transform(
                adapt_interval_f<Grid>{grid}, meta::rename<tuple, typename MssComponents::loop_intervals_t>());
        }

        template <class MssComponentsList, class NeedPositionals, class Grid, class DataStortes>
        auto make_mss_loops(Grid const &grid, DataStortes &data_stores) {
            return tuple_util::transform(
                [&](auto mss_components) {
                    using mss_components_t = decltype(mss_components);
                    using extent_t = get_extent_from_loop_intervals<typename mss_components_t::loop_intervals_t>;
                    using execution_t = typename mss_components_t::execution_engine_t;

                    auto composite =
                        make_composite<typename mss_components_t::mss_descriptor_t, NeedPositionals>(grid, data_stores);
                    auto loop_intervals = make_loop_intervals<mss_components_t>(grid);

                    return make_mss_loop<extent_t>(grid.k_total_length(),
                        execute::step<execution_t>,
                        std::move(composite),
                        std::move(loop_intervals));
                },
                MssComponentsList());
        }

        template <class IsStateful, class Grid, class Msses>
        auto make_intermediate(backend, IsStateful, Grid const &grid, Msses) {
            return [grid](auto external_data_stores) {
                auto alloc = sid::make_cached_allocator(&std::make_unique<char[]>);

                auto data_stores = hymap::concat(shift_origin_and_block(grid, std::move(external_data_stores)),
                    make_temporaries<Msses>(grid, alloc));

                using esfs_t = meta::flatten<meta::transform<get_esfs, Msses>>;

                using extent_map_t = get_extent_map<esfs_t>;

                using mss_components_array_t =
                    build_mss_components_array<Msses, extent_map_t, typename Grid::axis_type>;

                auto mss_loops = make_mss_loops<mss_components_array_t, IsStateful>(grid, data_stores);

                int_t total_i = grid.i_size();
                int_t total_j = grid.j_size();

                int_t NBI = (total_i + GT_DEFAULT_TILE_I - 1) / GT_DEFAULT_TILE_I;
                int_t NBJ = (total_j + GT_DEFAULT_TILE_J - 1) / GT_DEFAULT_TILE_J;

#pragma omp parallel for collapse(2)
                for (int_t bi = 0; bi < NBI; ++bi) {
                    for (int_t bj = 0; bj < NBJ; ++bj) {
                        int_t i_size = bi + 1 == NBI ? total_i - bi * GT_DEFAULT_TILE_I : GT_DEFAULT_TILE_I;
                        int_t j_size = bj + 1 == NBJ ? total_j - bj * GT_DEFAULT_TILE_J : GT_DEFAULT_TILE_J;
                        tuple_util::for_each([=](auto &&fun) { fun(bi, bj, i_size, j_size); }, mss_loops);
                    }
                }
            };
        }
    } // namespace x86
} // namespace gridtools
