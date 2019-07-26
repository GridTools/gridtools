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
#include "../stage_matrix.hpp"
#include "../stages_maker.hpp"
#include "fill_flush.hpp"
#include "fused_mss_loop_cuda.hpp"
#include "ij_cache.hpp"
#include "k_cache.hpp"
#include "launch_kernel.hpp"
#include "loop_interval.hpp"
#include "need_sync.hpp"
#include "shared_allocator.hpp"
#include "tmp_storage_sid.hpp"

#ifndef GT_DEFAULT_TILE_I
#define GT_DEFAULT_TILE_I 32
#endif
#ifndef GT_DEFAULT_TILE_J
#define GT_DEFAULT_TILE_J 8
#endif

namespace gridtools {
    namespace cuda {
        template <class PlhInfo, class = void>
        struct is_not_locally_cached : std::true_type {};

        template <class PlhInfo>
        struct is_not_locally_cached<PlhInfo,
            std::enable_if_t<PlhInfo::cache_io_policy_t::value == cache_io_policy::local>> : std::false_type {};

        template <class PlhInfo, class = void>
        struct is_ij_cached : std::false_type {};

        template <class PlhInfo>
        struct is_ij_cached<PlhInfo, std::enable_if_t<PlhInfo::cache_t::value == cache_type::ij>> : std::true_type {};

        template <class Msses, class Grid, class Allocator>
        auto make_temporaries(Grid const &grid, Allocator &allocator) {
            using plh_map_t = meta::filter<is_not_locally_cached, typename Msses::tmp_plh_map_t>;
            using extent_t = meta::rename<enclosing_extent, meta::transform<stage_matrix::get_extent, plh_map_t>>;
            return tuple_util::transform(
                [&allocator,
                    n_blocks_i = (grid.i_size() + GT_DEFAULT_TILE_I - 1) / GT_DEFAULT_TILE_I,
                    n_blocks_j = (grid.j_size() + GT_DEFAULT_TILE_J - 1) / GT_DEFAULT_TILE_J,
                    k_size = grid.k_size()](auto info) {
                    using info_t = decltype(info);
                    return make_tmp_storage<typename info_t::data_t>(typename info_t::num_colors_t(),
                        integral_constant<int_t, GT_DEFAULT_TILE_I>(),
                        integral_constant<int_t, GT_DEFAULT_TILE_J>(),
                        extent_t(),
                        n_blocks_i,
                        n_blocks_j,
                        k_size,
                        allocator);
                },
                hymap::from_keys_values<meta::transform<stage_matrix::get_plh, plh_map_t>, plh_map_t>());
        }

        template <class Plhs, template <class...> class ToKey, class Src>
        auto filter_map(Src &src) {
            return tuple_util::transform([&](auto plh) -> decltype(auto) { return at_key<decltype(plh)>(src); },
                hymap::from_keys_values<meta::transform<ToKey, Plhs>, Plhs>());
        }

        template <class Mss>
        auto make_ij_cached(shared_allocator &allocator) {
            using plh_map_t = meta::filter<is_ij_cached, typename Mss::plh_map_t>;
            using extent_t = meta::rename<enclosing_extent, meta::transform<stage_matrix::get_extent, plh_map_t>>;
            return tuple_util::transform(
                [&](auto info) {
                    using info_t = decltype(info);
                    return make_ij_cache<typename info_t::data_t>(typename info_t::num_colors_t(),
                        integral_constant<int_t, GT_DEFAULT_TILE_I>{},
                        integral_constant<int_t, GT_DEFAULT_TILE_J>{},
                        extent_t(),
                        allocator);
                },
                hymap::from_keys_values<meta::transform<stage_matrix::get_plh, plh_map_t>, plh_map_t>());
        }

        template <class Caches>
        using is_cached = meta::curry<meta::st_contains, meta::transform<cache_parameter, Caches>>;

        template <class Caches>
        using is_not_cached = meta::not_<is_cached<Caches>::template apply>;

        template <class DataStoreMap>
        auto block(DataStoreMap data_stores) {
            using block_map_t = hymap::keys<dim::i, dim::j>::values<integral_constant<int_t, GT_DEFAULT_TILE_I>,
                integral_constant<int_t, GT_DEFAULT_TILE_J>>;
            return tuple_util::transform(
                [](auto &&src) { return sid::block(std::forward<decltype(src)>(src), block_map_t{}); },
                std::move(data_stores));
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

        template <class Mss, class Grid>
        auto make_loop_intervals(Grid const &grid) {
            using deref_t = deref_f<compute_readonly_args<typename Mss::esf_sequence_t>>;
            using res_t = meta::rename<tuple,
                order_loop_intervals<typename Mss::execution_engine_t,
                    gridtools::make_loop_intervals<stages_maker<Mss>::template apply, typename Grid::interval_t>>>;
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
                    [&](auto fun) {
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

        template <int_t BlockSize>
        std::enable_if_t<BlockSize == 0, int_t> blocks_required_z(int_t) {
            return 1;
        }

        template <int_t BlockSize>
        std::enable_if_t<BlockSize != 0, int_t> blocks_required_z(int_t nz) {
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
                    blocks_required_z<BlockSize>(grid.k_size()),
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

        struct dummy_kernel_f {
            template <class Validator>
            GT_FUNCTION_DEVICE void operator()(int_t i_block, int_t j_block, Validator validator) const {}
        };

        template <class Mss, class Grid, class DataStores>
        auto make_mss_kernel(Grid const &grid, DataStores &data_stores) {
            /*
                        using k_caches_t = meta::filter<is_k_cache, caches_t>;
                        using non_local_k_caches_t = meta::filter<meta::not_<is_local_cache>::apply, k_caches_t>;

                        using plhs_t = extract_placeholders_from_mss<Mss>;

                        using k_cached_plhs_t = meta::filter<is_cached<k_caches_t>::template apply, plhs_t>;

                        using non_cached_plhs_t = meta::filter<is_not_cached<caches_t>::template apply, plhs_t>;
                        using cached_plhs_t = meta::filter<is_cached<non_local_k_caches_t>::template apply, plhs_t>;
                        */

            shared_allocator shared_alloc;

            auto composite = hymap::concat(sid::composite::keys<>::values<>(),
                //                make_k_cached_sids<Mss>(),
                make_ij_cached<Mss>(shared_alloc) //,
                //                filter_map<non_cached_plhs_t, meta::id>(data_stores),
                //                filter_map<cached_plhs_t, k_cache_original>(data_stores),
                //                make_bound_checkers<Mss>(data_stores)
            );

            //            auto loop_intervals = add_fill_flush_stages<Mss>(make_loop_intervals<Mss>(grid));
            //            auto kernel_fun = make_kernel_fun<Mss, DataStores>(grid, composite,
            //            std::move(loop_intervals));

            dummy_kernel_f kernel_fun;

            return kernel<typename Mss::extent_t,
                execute::block_size<typename Mss::execution_t>::value,
                decltype(kernel_fun)>{std::move(kernel_fun), shared_alloc.size()};
        }

        template <template <class...> class L, class Grid, class DataStores, class PrevKernel = no_kernel>
        void launch_msses(L<>, Grid const &grid, DataStores &&, PrevKernel prev_kernel = {}) {
            std::move(prev_kernel).launch_or_fuse(grid, no_kernel());
        }

        template <template <class...> class L,
            class Mss,
            class... Msses,
            class Grid,
            class DataStores,
            class PrevKernel = no_kernel>
        void launch_msses(L<Mss, Msses...>, Grid const &grid, DataStores &data_stores, PrevKernel prev_kernel = {}) {
            auto kernel = make_mss_kernel<Mss>(grid, data_stores);
            auto fused_kernel = std::move(prev_kernel).launch_or_fuse(grid, std::move(kernel));
            launch_msses(L<Msses...>(), grid, data_stores, std::move(fused_kernel));
        }

        template <class Spec, class Grid, class DataStores>
        void gridtools_backend_entry_point(backend, Spec, Grid const &grid, DataStores external_data_stores) {
            using msses_t = stage_matrix::make_fused_view<Spec>;
            auto cuda_alloc = sid::device::make_cached_allocator(&cuda_util::cuda_malloc<char>);
            auto data_stores =
                hymap::concat(block(std::move(external_data_stores)), make_temporaries<msses_t>(grid, cuda_alloc));
            launch_msses(meta::rename<meta::list, msses_t>(), grid, data_stores);
        }
    } // namespace cuda
} // namespace gridtools
