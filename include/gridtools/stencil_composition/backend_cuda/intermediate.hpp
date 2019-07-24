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
#include <utility>

#include "../../common/cuda_type_traits.hpp"
#include "../../common/cuda_util.hpp"
#include "../../common/defs.hpp"
#include "../../common/hymap.hpp"
#include "../../common/integral_constant.hpp"
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
#include "basic_token_execution_cuda.hpp"
#include "fill_flush.hpp"
#include "fused_mss_loop_cuda.hpp"
#include "ij_cache.hpp"
#include "k_cache.hpp"
#include "launch_kernel.hpp"
#include "loop_interval.hpp"
#include "need_sync.hpp"
#include "shared_allocator.hpp"
#include "tmp_storage_sid.hpp"

namespace gridtools {
    namespace cuda {
        template <class Mss>
        using get_esfs = typename Mss::esf_sequence_t;

        template <class Mss>
        using get_caches = typename Mss::cache_sequence_t;

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

        template <class Msses, class Grid, class Allocator>
        auto make_temporaries(Grid const &grid, Allocator &allocator) {
            using plhs_t = meta::filter<is_not_locally_cached_in_msses_f<Msses>::template apply,
                meta::filter<is_tmp_arg, extract_placeholders_from_msses<Msses>>>;
            using extent_map_t = get_extent_map<meta::flatten<meta::transform<get_esfs, Msses>>>;
            using extent_t = meta::rename<enclosing_extent,
                meta::transform<lookup_extent_map_f<extent_map_t>::template apply, plhs_t>>;
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
                hymap::from_keys_values<plhs_t, plhs_t>());
        }

        template <class Plhs, template <class...> class ToKey, class Src>
        auto filter_map(Src &src) {
            return tuple_util::transform([&](auto plh) -> decltype(auto) { return at_key<decltype(plh)>(src); },
                hymap::from_keys_values<meta::transform<ToKey, Plhs>, Plhs>());
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
                        extent_t(),
                        allocator);
                },
                hymap::from_keys_values<Plhs, Plhs>());
        }

        template <class Caches>
        using is_cached = meta::curry<meta::st_contains, meta::transform<cache_parameter, Caches>>;

        template <class Caches>
        using is_not_cached = meta::not_<is_cached<Caches>::template apply>;

        using block_map_t = hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, GT_DEFAULT_TILE_I>,
            integral_constant<int_t, GT_DEFAULT_TILE_J>>;

        template <class Grid, class DataStoreMap>
        auto shift_origin_and_block(Grid const &grid, DataStoreMap data_stores) {
            return tuple_util::transform(
                [offsets =
                        tuple_util::make<hymap::keys<dim::i, dim::j>::values>(grid.i_low_bound(), grid.j_low_bound())](
                    auto &src) { return sid::block(sid::shift_sid_origin(std::ref(src), offsets), block_map_t{}); },
                std::move(data_stores));
        }

        template <class Grid>
        auto make_positionals(meta::list<dim::i, dim::j, dim::k>, Grid const &grid) {
            using positionals_t = std::tuple<positional<dim::i>, positional<dim::j>, positional<dim::k>>;
            return tuple_util::transform([](auto pos) { return sid::block(pos, block_map_t{}); },
                hymap::convert_to<hymap::keys, positionals_t>(
                    positionals_t{grid.i_low_bound(), grid.j_low_bound(), 0}));
        }

        template <class Grid>
        tuple<> make_positionals(meta::list<>, Grid const &) {
            return {};
        }

        template <class Grid>
        hymap::keys<positional<dim::k>>::values<positional<dim::k>> make_positionals(meta::list<dim::k>, Grid const &) {
            return {};
        }

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

        template <class Deref, class NeedSync, class Extent, class... Stages>
        struct cuda_stage {
            template <class Ptr, class Strides, class Validator>
            GT_FUNCTION_DEVICE void operator()(
                Ptr const &GT_RESTRICT ptr, Strides const &GT_RESTRICT strides, Validator const &validator) const {
                syncthreads(NeedSync());
                if (validator(Extent()))
                    (void)(int[]){(Stages()(ptr, strides), 0)...};
            }
        };

        template <class Deref>
        struct adapt_stage_f {
            template <class Stage, class NeedSync>
            using apply = cuda_stage<Deref, typename NeedSync::type, typename Stage::extent_t, Stage>;
        };

        namespace lazy {
            template <class State, class Current>
            struct fuse_stages_folder : meta::lazy::push_front<State, Current> {};

            template <class Deref, class NeedSync, class Extent, class... Stages, class... CudaStages, class Stage>
            struct fuse_stages_folder<meta::list<cuda_stage<Deref, NeedSync, Extent, Stages...>, CudaStages...>,
                cuda_stage<Deref, std::false_type, Extent, Stage>> {
                using type = meta::list<cuda_stage<Deref, NeedSync, Extent, Stages..., Stage>, CudaStages...>;
            };

        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(fuse_stages_folder, (class State, class Current), (State, Current));

        template <class Stage>
        using get_esf = typename Stage::esf_t;

        template <class Mss, class ExtentMap, class Grid>
        auto make_loop_intervals(Grid const &grid) {
            using deref_t = deref_f<compute_readonly_args<typename Mss::esf_sequence_t>>;
            using default_interval_t = interval<typename Grid::axis_type::FromLevel,
                index_to_level<typename level_to_index<typename Grid::axis_type::ToLevel>::prior>>;
            using res_t = meta::rename<tuple,
                order_loop_intervals<typename Mss::execution_engine_t,
                    gridtools::make_loop_intervals<stages_maker<Mss, ExtentMap>::template apply, default_interval_t>>>;
            return tuple_util::transform(
                [&](auto interval) {
                    using interval_t = decltype(interval);
                    auto count = grid.count(meta::first<interval_t>{}, meta::second<interval_t>{});
                    using stages_t = meta::third<interval_t>;
                    using esfs_t = meta::transform<get_esf, stages_t>;
                    using need_syncs_t = need_sync<esfs_t, typename Mss::cache_sequence_t>;
                    using cuda_stages_t =
                        meta::transform<adapt_stage_f<deref_t>::template apply, stages_t, need_syncs_t>;
                    using fused_stages_t = meta::reverse<meta::lfold<fuse_stages_folder, meta::list<>, cuda_stages_t>>;

                    return make_loop_interval(count, fused_stages_t());
                },
                res_t{});
        }

        template <class... Funs>
        struct multi_kernel {
            tuple<Funs...> m_funs;

            template <class Validator>
            GT_FUNCTION_DEVICE void operator()(int_t i_block, int_t j_block, Validator const &validator) const {
                tuple_util::device::for_each(
                    [&](auto fun) GT_FORCE_INLINE_LAMBDA {
                        __threadfence_block();
                        fun(i_block, j_block, validator);
                    },
                    m_funs);
            }
        };

        template <class Fun>
        Fun make_multi_kernel(tuple<Fun> tup) {
            return tuple_util::get<0>(std::move(tup));
        }

        template <class... Funs>
        multi_kernel<Funs...> make_multi_kernel(tuple<Funs...> tup) {
            return {std::move(tup)};
        }

        template <uint_t BlockSize>
        std::enable_if_t<BlockSize == 0, int_t> blocks_required_z(uint_t) {
            return 1;
        }

        template <uint_t BlockSize>
        std::enable_if_t<BlockSize != 0, int_t> blocks_required_z(uint_t nz) {
            return (nz + BlockSize - 1) / BlockSize;
        }

        template <class MaxExtent, uint_t BlockSize, class... Funs>
        struct kernel {
            tuple<Funs...> m_funs;
            size_t m_shared_memory_size;

            template <class Grid, class Kernel>
            Kernel launch_or_fuse(Grid const &grid, Kernel kernel) && {
                launch_kernel<MaxExtent, GT_DEFAULT_TILE_I, GT_DEFAULT_TILE_J>(grid.i_size(),
                    grid.j_size(),
                    blocks_required_z<BlockSize>(grid.k_total_length()),
                    make_multi_kernel(std::move(m_funs)),
                    m_shared_memory_size);
                return kernel;
            }

            template <class Grid, class Fun>
            kernel<MaxExtent, BlockSize, Funs..., Fun> launch_or_fuse(
                Grid const &grid, kernel<MaxExtent, BlockSize, Fun> kernel) && {
                return {tuple_util::push_back(std::move(m_funs), tuple_util::get<0>(std::move(kernel.m_funs))),
                    std::max(m_shared_memory_size, kernel.m_shared_memory_size)};
            }
        };

        struct no_kernel {
            template <class Grid, class Kernel>
            Kernel launch_or_fuse(Grid &&, Kernel kernel) && {
                return kernel;
            }
        };

        template <class MaxExtent, class ExecutionType, class Fun>
        kernel<MaxExtent, 0, Fun> make_kernel(ExecutionType, Fun fun, size_t m_shared_memory_size) {
            return {{std::forward<Fun>(fun)}, m_shared_memory_size};
        }

        template <class MaxExtent, uint_t BlockSize, class Fun>
        kernel<MaxExtent, BlockSize, Fun> make_kernel(
            execute::parallel_block<BlockSize>, Fun fun, size_t m_shared_memory_size) {
            return {{std::forward<Fun>(fun)}, m_shared_memory_size};
        }

        template <bool NeedPositionals, class Mss, class Grid, class DataStores>
        auto make_mss_kernel(Grid const &grid, DataStores &data_stores) {
            using esfs_t = get_esfs<Mss>;
            using extent_map_t = get_extent_map<esfs_t>;

            using caches_t = get_caches<Mss>;
            using ij_caches_t = meta::filter<is_ij_cache, caches_t>;
            using k_caches_t = meta::filter<is_k_cache, caches_t>;
            using non_local_k_caches_t = meta::filter<meta::not_<is_local_cache>::apply, k_caches_t>;

            using plhs_t = extract_placeholders_from_mss<Mss>;

            using k_cached_plhs_t = meta::filter<is_cached<k_caches_t>::template apply, plhs_t>;
            using ij_cached_plhs_t = meta::filter<is_cached<ij_caches_t>::template apply, plhs_t>;

            using non_cached_plhs_t = meta::filter<is_not_cached<caches_t>::template apply, plhs_t>;
            using cached_plhs_t = meta::filter<is_cached<non_local_k_caches_t>::template apply, plhs_t>;

            shared_allocator shared_alloc;

            using positionals_t = meta::if_c<NeedPositionals,
                meta::list<dim::i, dim::j, dim::k>,
                meta::if_<meta::is_empty<non_local_k_caches_t>, meta::list<>, meta::list<dim::k>>>;

            auto composite = hymap::concat(sid::composite::keys<>::values<>(),
                make_k_cached_sids<Mss>(),
                make_ij_cached<ij_cached_plhs_t, extent_map_t>(shared_alloc),
                filter_map<non_cached_plhs_t, meta::id>(data_stores),
                filter_map<cached_plhs_t, k_cache_original>(data_stores),
                make_bound_checkers<Mss>(data_stores),
                make_positionals(positionals_t(), grid));

            auto loop_intervals = add_fill_flush_stages<Mss>(make_loop_intervals<Mss, extent_map_t>(grid));

            auto k_loop = make_k_loop<Mss>(grid, std::move(loop_intervals));

            auto kernel_fun = make_kernel_fun(composite, std::move(k_loop));

            using max_extent_t =
                meta::rename<enclosing_extent, meta::transform<get_esf_extent_f<extent_map_t>::template apply, esfs_t>>;
            return make_kernel<max_extent_t>(
                typename Mss::execution_engine_t(), std::move(kernel_fun), shared_alloc.size());
        }

        template <bool NeedPositionals,
            template <class...> class L,
            class Grid,
            class DataStores,
            class PrevKernel = no_kernel>
        void launch_msses(L<>, Grid const &grid, DataStores &data_stores, PrevKernel prev_kernel = {}) {
            std::move(prev_kernel).launch_or_fuse(grid, no_kernel());
        }

        template <bool NeedPositionals,
            template <class...> class L,
            class Mss,
            class... Msses,
            class Grid,
            class DataStores,
            class PrevKernel = no_kernel>
        void launch_msses(L<Mss, Msses...>, Grid const &grid, DataStores &data_stores, PrevKernel prev_kernel = {}) {
            auto kernel = make_mss_kernel<NeedPositionals, Mss>(grid, data_stores);
            auto fused_kernel = std::move(prev_kernel).launch_or_fuse(grid, std::move(kernel));
            launch_msses<NeedPositionals>(L<Msses...>(), grid, data_stores, std::move(fused_kernel));
        }

        template <class NeedPositionals, class Grid, class Msses>
        auto make_intermediate(backend, NeedPositionals, Grid const &grid, Msses msses) {
            return [grid](auto external_data_stores) {
                auto cuda_alloc = sid::device::make_cached_allocator(&cuda_util::cuda_malloc<char>);
                auto data_stores = hymap::concat(shift_origin_and_block(grid, std::move(external_data_stores)),
                    make_temporaries<Msses>(grid, cuda_alloc));
                launch_msses<NeedPositionals::value>(Msses(), grid, data_stores);
            };
        }
    } // namespace cuda
} // namespace gridtools
