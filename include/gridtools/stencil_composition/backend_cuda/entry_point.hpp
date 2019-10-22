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
#include "../mss.hpp"
#include "../positional.hpp"
#include "../sid/allocator.hpp"
#include "../sid/as_const.hpp"
#include "../sid/block.hpp"
#include "../sid/composite.hpp"
#include "../sid/concept.hpp"
#include "../sid/sid_shift_origin.hpp"
#include "../stage_matrix.hpp"
#include "fill_flush.hpp"
#include "fused_mss_loop_cuda.hpp"
#include "ij_cache.hpp"
#include "k_cache.hpp"
#include "launch_kernel.hpp"
#include "shared_allocator.hpp"
#include "tmp_storage_sid.hpp"

namespace gridtools {
    namespace cuda {
        template <class PlhInfo>
        using is_not_cached = meta::is_empty<typename PlhInfo::caches_t>;

        template <class Msses, class Backend, class Grid, class Allocator>
        auto make_temporaries(Backend be, Grid const &grid, Allocator &allocator) {
            using plh_map_t = meta::filter<is_not_cached, typename Msses::tmp_plh_map_t>;
            using extent_t = meta::rename<enclosing_extent, meta::transform<stage_matrix::get_extent, plh_map_t>>;
            return tuple_util::transform(
                [&allocator,
                    be,
                    n_blocks_i = (grid.i_size() + be.i_block_size() - 1) / be.i_block_size(),
                    n_blocks_j = (grid.j_size() + be.j_block_size() - 1) / be.j_block_size(),
                    k_size = grid.k_size()](auto info) {
                    using info_t = decltype(info);
                    return make_tmp_storage<typename info_t::data_t>(typename info_t::num_colors_t(),
                        be.i_block_size(),
                        be.j_block_size(),
                        extent_t(),
                        n_blocks_i,
                        n_blocks_j,
                        k_size,
                        allocator);
                },
                hymap::from_keys_values<meta::transform<stage_matrix::get_plh, plh_map_t>, plh_map_t>());
        }

        template <class Backend, class DataStoreMap>
        auto block(Backend be, DataStoreMap data_stores) {
            return tuple_util::transform(
                [=](auto &&src) {
                    return sid::block(std::forward<decltype(src)>(src),
                        tuple_util::make<hymap::keys<dim::i, dim::j>::values>(be.i_block_size(), be.j_block_size()));
                },
                std::move(data_stores));
        }

        template <class Is, class... Funs>
        struct multi_kernel;

        template <class... Funs, size_t... Is>
        struct multi_kernel<std::index_sequence<Is...>, Funs...> {
            tuple<Funs...> m_funs;

            template <class Validator>
            GT_FUNCTION_DEVICE void operator()(int_t i_block, int_t j_block, Validator const &validator) const {
                (void)(int[]){(tuple_util::device::get<Is>(m_funs)(i_block, j_block, validator), 0)...};
            }
        };

        template <class Fun>
        Fun make_multi_kernel(tuple<Fun> tup) {
            return tuple_util::get<0>(std::move(tup));
        }

        template <class... Funs>
        multi_kernel<std::index_sequence_for<Funs...>, Funs...> make_multi_kernel(tuple<Funs...> tup) {
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

        struct dummy_info {
            using extent_t = extent<>;
        };

        template <class PlhMap>
        struct get_extent_f {
            template <class Key>
            using apply = typename meta::mp_find<PlhMap, Key, dummy_info>::extent_t;
        };

        template <class MaxExtent, class PlhMap, uint_t BlockSize, class... Funs>
        struct kernel {
            tuple<Funs...> m_funs;
            size_t m_shared_memory_size;

            template <class Backend, class Grid, class Kernel>
            Kernel launch_or_fuse(Backend, Grid const &grid, Kernel kernel) && {
                launch_kernel<MaxExtent, Backend::i_block_size_t::value, Backend::j_block_size_t::value>(grid.i_size(),
                    grid.j_size(),
                    blocks_required_z<BlockSize>(grid.k_size()),
                    make_multi_kernel(std::move(m_funs)),
                    m_shared_memory_size);
                return kernel;
            }

            template <class OtherPlhMap,
                class Backend,
                class Grid,
                class Fun,
                class OutKeys = meta::transform<stage_matrix::get_key,
                    meta::filter<meta::not_<stage_matrix::get_is_const>::apply, PlhMap>>,
                class Extents = meta::transform<get_extent_f<OtherPlhMap>::template apply, OutKeys>,
                class Extent = meta::rename<enclosing_extent, Extents>,
                std::enable_if_t<Extent::iminus::value == 0 && Extent::iplus::value == 0 &&
                                     Extent::jminus::value == 0 && Extent::jplus::value == 0,
                    int> = 0>
            kernel<MaxExtent, stage_matrix::merge_plh_maps<PlhMap, OtherPlhMap>, BlockSize, Funs..., Fun>
            launch_or_fuse(Backend, Grid const &grid, kernel<MaxExtent, OtherPlhMap, BlockSize, Fun> kernel) && {
                return {tuple_util::push_back(std::move(m_funs), tuple_util::get<0>(std::move(kernel.m_funs))),
                    std::max(m_shared_memory_size, kernel.m_shared_memory_size)};
            }
        };

        struct no_kernel {
            template <class Backend, class Grid, class Kernel>
            Kernel launch_or_fuse(Backend, Grid &&, Kernel kernel) && {
                return kernel;
            }
        };

        template <class Backend, class Deref, class Mss, class Grid, class DataStores>
        auto make_mss_kernel(Grid const &grid, DataStores &data_stores) {
            shared_allocator shared_alloc;

            using ij_caches_t = meta::list<integral_constant<cache_type, cache_type::ij>>;
            using k_caches_t = meta::list<integral_constant<cache_type, cache_type::k>>;
            using no_caches_t = meta::list<>;

            using plh_map_t = typename Mss::plh_map_t;
            using keys_t = meta::rename<sid::composite::keys, meta::transform<meta::first, plh_map_t>>;

            auto composite = tuple_util::convert_to<keys_t::template values>(tuple_util::transform(
                overload(
                    [&](ij_caches_t, auto info) {
                        return make_ij_cache<decltype(info.data())>(info.num_colors(),
                            Backend::i_block_size(),
                            Backend::j_block_size(),
                            info.extent(),
                            shared_alloc);
                    },
                    [](k_caches_t, auto) { return k_cache_sid_t(); },
                    [&](no_caches_t, auto info) {
                        return sid::add_const(info.is_const(), at_key<decltype(info.plh())>(data_stores));
                    }),
                meta::transform<stage_matrix::get_caches, plh_map_t>(),
                plh_map_t()));

            auto kernel_fun = make_kernel_fun<Deref, Mss, Backend::k_block_size()>(grid, composite);

            return kernel<typename Mss::extent_t,
                typename Mss::plh_map_t,
                (execute::is_parallel<typename Mss::execution_t>{} ? Backend::k_block_size() : 0),
                decltype(kernel_fun)>{std::move(kernel_fun), shared_alloc.size()};
        }

        template <class Deref,
            class Backend,
            template <class...> class L,
            class Grid,
            class DataStores,
            class PrevKernel = no_kernel>
        void launch_msses(Backend be, L<>, Grid const &grid, DataStores &&, PrevKernel prev_kernel = {}) {
            std::move(prev_kernel).launch_or_fuse(be, grid, no_kernel());
        }

        template <class Deref,
            class Backend,
            template <class...> class L,
            class Mss,
            class... Msses,
            class Grid,
            class DataStores,
            class PrevKernel = no_kernel>
        void launch_msses(
            Backend be, L<Mss, Msses...>, Grid const &grid, DataStores &data_stores, PrevKernel prev_kernel = {}) {
            auto kernel = make_mss_kernel<Backend, Deref, Mss>(grid, data_stores);
            auto fused_kernel = std::move(prev_kernel).launch_or_fuse(be, grid, std::move(kernel));
            launch_msses<Deref>(be, L<Msses...>(), grid, data_stores, std::move(fused_kernel));
        }

        template <class Keys>
        struct deref_f {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
            template <class Key, class T>
            GT_FUNCTION std::enable_if_t<is_texture_type<T>::value && meta::st_contains<Keys, Key>::value, T>
            operator()(Key, T const *ptr) const {
                return __ldg(ptr);
            }
#endif
            template <class Key, class Ptr>
            GT_FUNCTION decltype(auto) operator()(Key, Ptr ptr) const {
                return *ptr;
            }
        };

        template <class Msses, class Backend, class Grid, class DataStores>
        void entry_point(Backend be, Grid const &grid, DataStores external_data_stores) {
            auto cuda_alloc = sid::device::make_cached_allocator(&cuda_util::cuda_malloc<char>);
            auto data_stores = hymap::concat(
                cuda::block(be, std::move(external_data_stores)), make_temporaries<Msses>(be, grid, cuda_alloc));
            using const_keys_t = meta::transform<stage_matrix::get_key,
                meta::filter<stage_matrix::get_is_const, typename Msses::plh_map_t>>;
            launch_msses<deref_f<const_keys_t>>(be, meta::rename<meta::list, Msses>(), grid, data_stores);
        }

        template <class... Params, class Spec, class Grid, class DataStores>
        void gridtools_backend_entry_point(backend<Params...> be, Spec, Grid const &grid, DataStores data_stores) {
            using new_spec_t = fill_flush::transform_spec<Spec>;
            using msses_t = stage_matrix::make_fused_view<new_spec_t>;
            entry_point<msses_t>(
                be, grid, fill_flush::transform_data_stores<typename msses_t::plh_map_t>(std::move(data_stores)));
        }
    } // namespace cuda
} // namespace gridtools
