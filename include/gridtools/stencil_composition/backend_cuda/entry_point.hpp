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
        using i_block_size_t = integral_constant<int_t, 32>;
        using j_block_size_t = integral_constant<int_t, 8>;

        template <class PlhInfo>
        using is_not_cached = meta::is_empty<typename PlhInfo::caches_t>;

        template <class Msses, class Grid, class Allocator>
        auto make_temporaries(Grid const &grid, Allocator &allocator) {
            using plh_map_t = meta::filter<is_not_cached, typename Msses::tmp_plh_map_t>;
            using extent_t = meta::rename<enclosing_extent, meta::transform<stage_matrix::get_extent, plh_map_t>>;
            return tuple_util::transform(
                [&allocator,
                    n_blocks_i = (grid.i_size() + i_block_size_t::value - 1) / i_block_size_t::value,
                    n_blocks_j = (grid.j_size() + j_block_size_t::value - 1) / j_block_size_t::value,
                    k_size = grid.k_size()](auto info) {
                    using info_t = decltype(info);
                    return make_tmp_storage<typename info_t::data_t>(typename info_t::num_colors_t(),
                        i_block_size_t(),
                        j_block_size_t(),
                        extent_t(),
                        n_blocks_i,
                        n_blocks_j,
                        k_size,
                        allocator);
                },
                hymap::from_keys_values<meta::transform<stage_matrix::get_plh, plh_map_t>, plh_map_t>());
        }

        template <class DataStoreMap>
        auto block(DataStoreMap data_stores) {
            return tuple_util::transform(
                [](auto &&src) {
                    return sid::block(std::forward<decltype(src)>(src),
                        hymap::keys<dim::i, dim::j>::values<i_block_size_t, j_block_size_t>());
                },
                std::move(data_stores));
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
                launch_kernel<MaxExtent, i_block_size_t::value, j_block_size_t::value>(grid.i_size(),
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

        template <class Mss, class Grid, class DataStores>
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
                        return make_ij_cache<decltype(info.data())>(
                            info.num_colors(), i_block_size_t(), j_block_size_t(), info.extent(), shared_alloc);
                    },
                    [](k_caches_t, auto) { return k_cache_sid_t(); },
                    [&](no_caches_t, auto info) {
                        return sid::add_const(info.is_const(), at_key<decltype(info.plh())>(data_stores));
                    }),
                meta::transform<stage_matrix::get_caches, plh_map_t>(),
                plh_map_t()));

            auto kernel_fun = make_kernel_fun<Mss>(grid, composite);

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
        void entry_point(Spec, Grid const &grid, DataStores external_data_stores) {
            using msses_t = stage_matrix::make_fused_view<Spec>;
            auto cuda_alloc = sid::device::make_cached_allocator(&cuda_util::cuda_malloc<char>);
            auto data_stores =
                hymap::concat(block(std::move(external_data_stores)), make_temporaries<msses_t>(grid, cuda_alloc));
            launch_msses(meta::rename<meta::list, msses_t>(), grid, data_stores);
        }

        template <class Spec, class Grid, class DataStores>
        void gridtools_backend_entry_point(backend, Spec, Grid const &grid, DataStores data_stores) {
            entry_point(fill_flush::transform_spec<Spec>(),
                grid,
                fill_flush::transform_data_stores(Spec(), std::move(data_stores)));
        }
    } // namespace cuda
} // namespace gridtools
