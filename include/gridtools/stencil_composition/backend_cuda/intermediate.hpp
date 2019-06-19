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

#include "../../common/cuda_type_traits.hpp"
#include "../../common/cuda_util.hpp"
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
#include "../make_loop_intervals.hpp"
#include "../mss.hpp"
#include "../positional.hpp"
#include "../sid/allocator.hpp"
#include "../sid/block.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "../stages_maker.hpp"
#include "fused_mss_loop_cuda.hpp"
#include "ij_cache.hpp"
#include "k_cache.hpp"
#include "launch_kernel.hpp"
#include "need_sync.hpp"
#include "shared_allocator.hpp"
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
        auto make_temporaries(Allocator &allocator, Grid const &grid) {
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
                hymap::from_keys_values<Plhs, Plhs>{});
        }

        template <class Plhs, class Src>
        auto filter_map(Src const &src) {
            return tuple_util::transform(
                [&](auto plh) -> decltype(auto) { return at_key<decltype(plh)>(src); }, Plhs{});
        }

        template <class Plhs, class ExtentMap>
        auto make_ij_cached(shared_allocator &allocator) {
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

        template <class Plhs, class Esfs>
        auto make_k_cached() {
            return tuple_util::transform([&](auto plh) { return make_k_cache(plh, Esfs()); }, Plhs{});
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

        template <class ExecutionType>
        std::enable_if_t<!execute::is_parallel<ExecutionType>::value, int_t> blocks_required_z(uint_t) {
            return 1;
        }

        template <class ExecutionType>
        std::enable_if_t<execute::is_parallel<ExecutionType>::value, int_t> blocks_required_z(uint_t nz) {
            return (nz + ExecutionType::block_size - 1) / ExecutionType::block_size;
        }

        template <class Caches>
        using is_cached = meta::curry<meta::st_contains, meta::transform<cache_parameter, Caches>>;

        template <class Caches>
        using is_not_cached = meta::not_<is_cached<Caches>::template apply>;

        template <class Mss>
        using get_esfs = typename Mss::esf_sequence_t;

        template <class Mss>
        using get_caches = typename Mss::cache_sequence_t;

        template <class CacheSequence, class Composite, class LowerKBounds, class UpperKBounds>
        auto make_local_domain(Composite &composite, LowerKBounds lower_k_bounds, UpperKBounds upper_k_bounds) {
            return local_domain<Composite, CacheSequence, LowerKBounds, UpperKBounds>{
                composite, std::move(lower_k_bounds), std::move(upper_k_bounds)};
        }

        template <class Plh>
        struct plh_is_not_locally_cached_for_mss_f {
            template <class Mss>
            using apply = negation<meta::
                    st_contains<meta::transform<cache_parameter, meta::filter<is_local_cache, get_caches<Mss>>>, Plh>>;
        };

        template <class Msses>
        struct is_not_locally_cached_in_msses_f {
            template <class Plh>
            using apply = meta::all_of<plh_is_not_locally_cached_for_mss_f<Plh>::template apply, Msses>;
        };

        GT_FUNCTION_DEVICE void syncthreads(std::true_type) { __syncthreads(); }
        GT_FUNCTION_DEVICE void syncthreads(std::false_type) {}

        template <class ReadOnlyArgs>
        struct deref_f {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
            template <class Arg, class T>
            GT_FUNCTION std::enable_if_t<meta::st_contains<ReadOnlyArgs, Arg>::value && is_texture_type<T>::value, T>
            operator()(T *ptr) const {
                return __ldg(ptr);
            }
#endif
            template <class, class Ptr>
            GT_FUNCTION decltype(auto) operator()(Ptr ptr) const {
                return *ptr;
            }
        };

        template <class Deref, class Stage, class NeedSync>
        struct cuda_stage_adaptor_f {
            template <class Ptr, class Strides, class Validator>
            GT_FUNCTION_DEVICE void operator()(
                Ptr const &GT_RESTRICT ptr, Strides const &GT_RESTRICT strides, Validator const &validator) const {
                if (validator(Stage::extent()))
                    Stage{}.template operator()<Deref>(ptr, strides);
                syncthreads(NeedSync{});
            }
        };

        template <class Count, class Stages>
        class cuda_loop_interval {
            Count m_count;
            Stages m_stages;

          public:
            cuda_loop_interval(Count count, Stages stages) : m_count(count), m_stages(stages) {}

            GT_FUNCTION_DEVICE Count count() const { return m_count; }

            template <class Ptr, class Strides, class Validator>
            GT_FUNCTION_DEVICE void operator()(
                Ptr const &GT_RESTRICT ptr, Strides const &GT_RESTRICT strides, Validator const &validator) const {
                tuple_util::device::for_each([&](auto const &stage) { stage(ptr, strides, validator); }, m_stages);
            }
        };

        template <class Stage>
        using get_esf = typename Stage::esf_t;

        template <class ExecutionType, class Mss, class ExtentMap, class Grid>
        auto make_loop_intervals(Grid const &grid) {
            using deref_t = deref_f<compute_readonly_args<typename Mss::esf_sequence_t>>;
            using default_interval_t = interval<typename Grid::axis_type::FromLevel,
                index_to_level<typename level_to_index<typename Grid::axis_type::ToLevel>::prior>>;
            using res_t = meta::rename<tuple,
                order_loop_intervals<ExecutionType,
                    gridtools::make_loop_intervals<stages_maker<Mss, ExtentMap>::template apply, default_interval_t>>>;
            return tuple_util::transform(
                [&](auto interval) {
                    using interval_t = decltype(interval);
                    auto count = grid.count(meta::first<interval_t>{}, meta::second<interval_t>{});
                    using stages_t = meta::third<interval_t>;
                    using esfs_t = meta::transform<get_esf, stages_t>;
                    using need_syncs_t = need_sync<esfs_t, typename Mss::cache_sequence_t>;
                    using adapted_stages_t = meta::rename<tuple,
                        meta::transform<meta::curry<cuda_stage_adaptor_f, deref_t>::template apply,
                            stages_t,
                            need_syncs_t>>;
                    return cuda_loop_interval<decltype(count), adapted_stages_t>(count, adapted_stages_t{});
                },
                res_t{});
        }

        template <class IsStateful, class Grid, class Msses>
        auto make_intermediate(backend, IsStateful, Grid const &grid, Msses) {
            return [grid](auto const &data_store_map) {
                using extent_map_t = get_extent_map<meta::flatten<meta::transform<get_esfs, Msses>>>;
                using all_tmp_plhs_t = meta::filter<is_not_locally_cached_in_msses_f<Msses>::template apply,
                    meta::filter<is_tmp_arg, extract_placeholders_from_msses<Msses>>>;
                auto cuda_alloc = sid::device::make_cached_allocator(&cuda_util::cuda_malloc<char>);
                auto temporaries = make_temporaries<all_tmp_plhs_t, extent_map_t>(cuda_alloc, grid);
                for_each<Msses>([&](auto mss) {
                    using mss_t = decltype(mss);
                    using esfs_t = get_esfs<mss_t>;

                    using caches_t = get_caches<mss_t>;
                    using ij_caches_t = meta::filter<is_ij_cache, caches_t>;
                    using local_caches_t = meta::filter<is_local_cache, caches_t>;
                    using k_caches_t = meta::filter<is_k_cache, caches_t>;
                    using non_local_k_caches_t = meta::filter<meta::not_<is_local_cache>::apply, k_caches_t>;

                    using plhs_t = extract_placeholders_from_mss<mss_t>;
                    using non_tmp_plhs_t = meta::filter<meta::not_<is_tmp_arg>::apply, plhs_t>;
                    using tmp_plhs_t = meta::filter<is_tmp_arg, plhs_t>;

                    using k_cached_plhs_t = meta::filter<is_cached<k_caches_t>::template apply, plhs_t>;
                    using ij_cached_plhs_t = meta::filter<is_cached<ij_caches_t>::template apply, plhs_t>;
                    using non_tmp_non_cached_plhs_t =
                        meta::filter<is_not_cached<caches_t>::template apply, non_tmp_plhs_t>;
                    using tmp_non_cached_plhs_t = meta::filter<is_not_cached<caches_t>::template apply, tmp_plhs_t>;
                    using non_tmp_k_cached_plhs_t =
                        meta::filter<is_cached<non_local_k_caches_t>::template apply, non_tmp_plhs_t>;
                    using tmp_k_cached_plhs_t =
                        meta::filter<is_cached<non_local_k_caches_t>::template apply, tmp_plhs_t>;

                    using non_cached_tmp_plhs_t =
                        meta::filter<is_not_cached<local_caches_t>::template apply, tmp_plhs_t>;

                    using composite_keys_t = meta::rename<sid::composite::keys,
                        meta::concat< //
                            k_cached_plhs_t,
                            ij_cached_plhs_t,
                            non_tmp_non_cached_plhs_t,
                            tmp_non_cached_plhs_t,
                            meta::transform<k_cache_original, non_tmp_k_cached_plhs_t>,
                            meta::transform<k_cache_original, tmp_k_cached_plhs_t>,
                            positionals_t>>;

                    shared_allocator shared_alloc;

                    auto composite = tuple_util::convert_to<composite_keys_t::template values>(
                        tuple_util::deep_copy(tuple_util::flatten(tuple_util::make<std::tuple>( //
                            make_k_cached<k_cached_plhs_t, esfs_t>(),
                            make_ij_cached<ij_cached_plhs_t, extent_map_t>(shared_alloc),
                            make_data_stores<non_tmp_non_cached_plhs_t>(grid, data_store_map),
                            filter_map<tmp_non_cached_plhs_t>(temporaries),
                            make_data_stores<non_tmp_k_cached_plhs_t>(grid, data_store_map),
                            filter_map<tmp_k_cached_plhs_t>(temporaries),
                            make_positionals(grid)))));

                    auto local_domain = make_local_domain<typename mss_t::cache_sequence_t>(composite,
                        make_k_lower_bounds<non_tmp_k_cached_plhs_t>(data_store_map),
                        make_k_upper_bounds<non_tmp_k_cached_plhs_t>(data_store_map));

                    using execution_type_t = typename mss_t::execution_engine_t;
                    using max_extent_t = meta::rename<enclosing_extent,
                        meta::transform<get_esf_extent_f<extent_map_t>::template apply, esfs_t>>;

                    auto loop_intervals = make_loop_intervals<execution_type_t, mss_t, extent_map_t>(grid);

                    // number of blocks required
                    uint_t xblocks = (grid.i_size() + GT_DEFAULT_TILE_I - 1) / GT_DEFAULT_TILE_I;
                    uint_t yblocks = (grid.j_size() + GT_DEFAULT_TILE_J - 1) / GT_DEFAULT_TILE_J;
                    uint_t zblocks = blocks_required_z<execution_type_t>(grid.k_total_length());

                    launch_kernel<max_extent_t, GT_DEFAULT_TILE_I, GT_DEFAULT_TILE_J>({xblocks, yblocks, zblocks},
                        make_kernel<execution_type_t, max_extent_t, esfs_t, GT_DEFAULT_TILE_I, GT_DEFAULT_TILE_J>(
                            local_domain, grid, loop_intervals),
                        shared_alloc.size());
                });
            };
        }
    } // namespace cuda
} // namespace gridtools
