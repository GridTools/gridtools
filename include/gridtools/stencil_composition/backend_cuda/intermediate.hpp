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

#include <tuple>
#include <type_traits>

#include "../../common/defs.hpp"
#include "../../common/hymap.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta.hpp"
#include "../arg.hpp"
#include "../caches/cache_traits.hpp"
#include "../compute_extents_metafunctions.hpp"
#include "../dim.hpp"
#include "../esf_metafunctions.hpp"
#include "../extent.hpp"
#include "../extract_placeholders.hpp"
#include "../grid.hpp"
#include "../local_domain.hpp"
#include "../mss_components.hpp"
#include "../mss_components_metafunctions.hpp"
#include "../positional.hpp"
#include "../sid/block.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "fused_mss_loop_cuda.hpp"
#include "ij_cache.hpp"
#include "launch_kernel.hpp"
#include "shared_allocator.hpp"
#include "simple_device_memory_allocator.hpp"
#include "tmp_storage_sid.hpp"

namespace gridtools {
    namespace cuda {
        template <class Composite, class CacheSequence, class KLowerBoundsMap, class KUpperBoundsMap>
        class local_domain {
            KLowerBoundsMap m_k_lower_bounds_map;
            KUpperBoundsMap m_k_upper_bounds_map;

            template <class Arg, std::enable_if_t<has_key<KLowerBoundsMap, Arg>::value, int> = 0>
            GT_FUNCTION_DEVICE bool left_validate_k_pos(int_t pos) const {
                return pos >= device::at_key<Arg>(m_k_lower_bounds_map);
            }
            template <class Arg, std::enable_if_t<!has_key<KLowerBoundsMap, Arg>::value, int> = 0>
            GT_FUNCTION_DEVICE bool left_validate_k_pos(int_t pos) const {
                return true;
            }

            template <class Arg, std::enable_if_t<has_key<KUpperBoundsMap, Arg>::value, int> = 0>
            GT_FUNCTION_DEVICE bool right_validate_k_pos(int_t pos) const {
                return pos < device::at_key<Arg>(m_k_upper_bounds_map);
            }
            template <class Arg, std::enable_if_t<!has_key<KUpperBoundsMap, Arg>::value, int> = 0>
            GT_FUNCTION_DEVICE bool right_validate_k_pos(int_t pos) const {
                return true;
            }

          public:
            using cache_sequence_t = CacheSequence;

            using ptr_holder_t = sid::ptr_holder_type<Composite>;
            using ptr_t = sid::ptr_type<Composite>;
            using strides_t = sid::strides_type<Composite>;
            using ptr_diff_t = sid::ptr_diff_type<Composite>;

            local_domain(Composite &composite, KLowerBoundsMap k_lower_bounds_map, KUpperBoundsMap k_upper_bounds_map)
                : m_k_lower_bounds_map(std::move(k_lower_bounds_map)),
                  m_k_upper_bounds_map(std::move(k_upper_bounds_map)), m_ptr_holder(sid::get_origin(composite)),
                  m_strides(sid::get_strides(composite)) {}

            template <class Arg>
            GT_FUNCTION_DEVICE bool validate_k_pos(int_t pos) const {
                return left_validate_k_pos<Arg>(pos) && right_validate_k_pos<Arg>(pos);
            }

            ptr_holder_t m_ptr_holder;
            strides_t m_strides;
        };

        template <class CacheSequence, class Composite, class LowerKBounds, class UpperKBounds>
        auto make_local_domain(Composite &composite, LowerKBounds lower_k_bounds, UpperKBounds upper_k_bounds) {
            return local_domain<Composite, CacheSequence, LowerKBounds, UpperKBounds>{
                composite, std::move(lower_k_bounds), std::move(upper_k_bounds)};
        }

        template <class Mss>
        using get_esfs = unwrap_independent<typename Mss::esf_sequence_t>;

        template <class Mss>
        struct non_cached_tmp_f {
            using local_caches_t = meta::filter<is_local_cache, typename Mss::cache_sequence_t>;
            using cached_args_t = meta::transform<cache_parameter, local_caches_t>;

            template <class Arg>
            using apply = bool_constant<is_tmp_arg<Arg>::value && !meta::st_contains<cached_args_t, Arg>::value>;
        };

        template <class Mss>
        struct ij_cached_tmp_f {
            using cached_args_t =
                meta::transform<cache_parameter, meta::filter<is_ij_cache, typename Mss::cache_sequence_t>>;

            template <class Arg>
            using apply = meta::st_contains<cached_args_t, Arg>;
        };

        template <class Mss>
        struct non_tmp_cached_f {
            using non_tmp_cached_args_t = meta::transform<cache_parameter,
                meta::filter<meta::not_<is_local_cache>::apply, typename Mss::cache_sequence_t>>;

            template <class Arg>
            using apply = meta::st_contains<non_tmp_cached_args_t, Arg>;
        };

        template <class Plhs, class ExtentMap, class Allocator, class Grid>
        auto make_non_cached_temporaries(Allocator &allocator, Grid const &grid) {
            using extent_t =
                meta::rename<enclosing_extent, meta::transform<lookup_extent_map_f<ExtentMap>::template apply, Plhs>>;
            return tuple_util::transform(
                [&allocator,
                    n_blocks_i = (grid.i_size() + GT_DEFAULT_TILE_I - 1) / GT_DEFAULT_TILE_I,
                    n_blocks_j = (grid.j_size() + GT_DEFAULT_TILE_J - 1) / GT_DEFAULT_TILE_J,
                    k_size = grid.k_max() + 1](auto plh) {
                    return make_tmp_storage(plh,
                        integral_constant<int_t, GT_DEFAULT_TILE_I>{},
                        integral_constant<int_t, GT_DEFAULT_TILE_J>{},
                        extent_t{},
                        n_blocks_i,
                        n_blocks_j,
                        k_size,
                        allocator);
                },
                Plhs{});
        }

        template <class Plhs, class ExtentMap>
        auto make_ij_cached_temporaries(shared_allocator &allocator) {
            using extent_t =
                meta::rename<enclosing_extent, meta::transform<lookup_extent_map_f<ExtentMap>::template apply, Plhs>>;
            return tuple_util::transform(
                [&](auto plh) {
                    return make_ij_cache(plh,
                        integral_constant<int_t, GT_DEFAULT_TILE_I>{},
                        integral_constant<int_t, GT_DEFAULT_TILE_J>{},
                        extent_t{},
                        allocator);
                },
                Plhs{});
        }

        template <class Plhs, class Grid, class DataStoreMap>
        auto make_data_stores(Grid const &grid, DataStoreMap const &data_store_map) {
            using block_map_t = hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, GT_DEFAULT_TILE_I>,
                integral_constant<int_t, GT_DEFAULT_TILE_J>>;
            return tuple_util::transform(
                [&data_store_map,
                    offsets = tuple_util::make<hymap::keys<dim::i, dim::j>::values>(
                        grid.i_low_bound(), grid.j_low_bound())](auto plh) {
                    using plh_t = decltype(plh);
                    return sid::block(
                        sid::shift_sid_origin(std::ref(at_key<plh_t>(data_store_map)), offsets), block_map_t{});
                },
                Plhs{});
        }

        using positionals_t = std::tuple<positional<dim::i>, positional<dim::j>, positional<dim::k>>;

        template <class Grid>
        positionals_t make_positionals(Grid const &grid) {
            return {grid.i_low_bound(), grid.j_low_bound(), 0};
        }

        template <class DataStoreMap>
        struct has_k_lower_bound_f {
            using map_t = hymap::to_meta_map<DataStoreMap>;

            template <class Plh>
            using apply =
                has_key<sid::lower_bounds_type<std::decay_t<meta::second<meta::mp_find<map_t, Plh>>>>, dim::k>;
        };

        template <class Plhs, class DataStoreMap>
        auto make_k_lower_bounds(DataStoreMap const &data_store_map) {
            using plhs_t = meta::filter<has_k_lower_bound_f<DataStoreMap>::template apply, Plhs>;
            return tuple_util::convert_to<meta::rename<hymap::keys, plhs_t>::template values>(tuple_util::transform(
                [&](auto plh) {
                    using plh_t = decltype(plh);
                    return at_key<dim::k>(sid::get_lower_bounds(at_key<plh_t>(data_store_map)));
                },
                plhs_t{}));
        }

        template <class DataStoreMap>
        struct has_k_upper_bound_f {
            using map_t = hymap::to_meta_map<DataStoreMap>;

            template <class Plh>
            using apply =
                has_key<sid::upper_bounds_type<std::decay_t<meta::second<meta::mp_find<map_t, Plh>>>>, dim::k>;
        };

        template <class Plhs, class DataStoreMap>
        auto make_k_upper_bounds(DataStoreMap const &data_store_map) {
            using plhs_t = meta::filter<has_k_upper_bound_f<DataStoreMap>::template apply, Plhs>;
            return tuple_util::convert_to<meta::rename<hymap::keys, plhs_t>::template values>(tuple_util::transform(
                [&](auto plh) {
                    using plh_t = decltype(plh);
                    return at_key<dim::k>(sid::get_upper_bounds(at_key<plh_t>(data_store_map)));
                },
                plhs_t{}));
        }

        template <class Grid, class Msses>
        class intermediate {
            GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);

            using esfs_t = meta::flatten<meta::transform<get_esfs, Msses>>;

            // First we need to compute the association between placeholders and extents.
            // This information is needed to allocate temporaries, and to provide the extent information to the
            // user.
            using extent_map_t = get_extent_map<esfs_t>;

            // member fields
            Grid m_grid;

          public:
            intermediate(Grid const &grid) : m_grid(grid) {}

            template <class DataStoreMap>
            void operator()(DataStoreMap const &data_store_map) const {
                for_each<Msses>([&](auto mss) {
                    using mss_t = decltype(mss);
                    using plhs_t = extract_placeholders_from_mss<mss_t>;
                    using tmp_plhs_t = meta::filter<is_tmp_arg, plhs_t>;
                    using non_tmp_plhs_t = meta::filter<meta::not_<is_tmp_arg>::apply, plhs_t>;
                    using non_cached_tmp_plhs_t = meta::filter<non_cached_tmp_f<mss_t>::template apply, tmp_plhs_t>;
                    using ij_cached_tmp_plhs_t = meta::filter<ij_cached_tmp_f<mss_t>::template apply, tmp_plhs_t>;
                    using non_tmp_cached_plhs_t = meta::filter<non_tmp_cached_f<mss_t>::apply, non_tmp_plhs_t>;

                    shared_allocator shared_alloc;
                    simple_device_memory_allocator cuda_allocator;

                    auto non_tmp_data_stores = make_data_stores<non_tmp_plhs_t>(m_grid, data_store_map);

                    auto non_cached_temporaries =
                        make_non_cached_temporaries<non_cached_tmp_plhs_t, extent_map_t>(cuda_allocator, m_grid);

                    auto ij_cached_temporaries =
                        make_ij_cached_temporaries<ij_cached_tmp_plhs_t, extent_map_t>(shared_alloc);

                    auto positionals = make_positionals(m_grid);

                    using keys_t = meta::rename<sid::composite::keys,
                        meta::concat<non_tmp_plhs_t, non_cached_tmp_plhs_t, ij_cached_tmp_plhs_t, positionals_t>>;

                    auto composite = tuple_util::convert_to<keys_t::template values>(
                        tuple_util::deep_copy(tuple_util::flatten(tuple_util::make<std::tuple>(
                            non_tmp_data_stores, non_cached_temporaries, ij_cached_temporaries, positionals))));

                    auto local_domain = make_local_domain<typename mss_t::cache_sequence_t>(composite,
                        make_k_lower_bounds<non_tmp_cached_plhs_t>(data_store_map),
                        make_k_upper_bounds<non_tmp_cached_plhs_t>(data_store_map));

                    using local_domain_t = decltype(local_domain);

                    using execution_type_t = typename mss_t::execution_engine_t;

                    using mss_components_t = mss_components<mss_t, extent_map_t, typename Grid::axis_type>;
                    using run_functor_args_t = run_functor_arguments<backend,
                        typename mss_components_t::linear_esf_t,
                        typename mss_components_t::loop_intervals_t,
                        execution_type_t>;
                    using max_extent_t = typename run_functor_args_t::max_extent_t;

                    // number of blocks required
                    uint_t xblocks = (m_grid.i_size() + GT_DEFAULT_TILE_I - 1) / GT_DEFAULT_TILE_I;
                    uint_t yblocks = (m_grid.j_size() + GT_DEFAULT_TILE_J - 1) / GT_DEFAULT_TILE_J;
                    uint_t zblocks =
                        fused_mss_loop_cuda_impl_::blocks_required_z<execution_type_t>(m_grid.k_total_length());

                    launch_kernel<max_extent_t, GT_DEFAULT_TILE_I, GT_DEFAULT_TILE_J>({xblocks, yblocks, zblocks},
                        make_kernel<run_functor_args_t, GT_DEFAULT_TILE_I, GT_DEFAULT_TILE_J>(local_domain, m_grid),
                        shared_alloc.size());
                });
            }
        };

        template <class IsStateful, class Grid, class Msses>
        intermediate<Grid, Msses> make_intermediate(backend, IsStateful, Grid const &grid, Msses) {
            return {grid};
        }
    } // namespace cuda
} // namespace gridtools
