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

#include "../mss_functor.hpp"

/**@file
 * @brief fused mss loop implementations for the mc backend
 */
namespace gridtools {
    namespace _impl {
        /**
         * @brief Meta function to check if an MSS can be executed in parallel along k-axis.
         */
        template <typename Mss>
        GT_META_DEFINE_ALIAS(is_mss_kparallel, execute::is_parallel, typename Mss::execution_engine_t);

        /**
         * @brief Meta function to check if all MSS in an MssComponents array can be executed in parallel along k-axis.
         */
        template <typename Msses>
        GT_META_DEFINE_ALIAS(all_mss_kparallel, meta::all_of, (is_mss_kparallel, Msses));
    } // namespace _impl

    /**
     * @brief loops over all blocks and execute sequentially all mss functors for each block
     * @tparam MssComponents a meta array with the mss components of all MSS
     */
    template <class MssComponents,
        class LocalDomainListArray,
        class Grid,
        enable_if_t<!_impl::all_mss_kparallel<MssComponents>::value, int> = 0>
    void fused_mss_loop(backend::mc, LocalDomainListArray const &local_domain_lists, const Grid &grid) {
        GT_STATIC_ASSERT((meta::all_of<is_mss_components, MssComponents>::value), GT_INTERNAL_ERROR);

        execinfo_mc exinfo(grid);
        const int_t i_blocks = exinfo.i_blocks();
        const int_t j_blocks = exinfo.j_blocks();
#pragma omp parallel for collapse(2)
        for (int_t bj = 0; bj < j_blocks; ++bj) {
            for (int_t bi = 0; bi < i_blocks; ++bi) {
                run_mss_functors<MssComponents>(backend::mc{}, local_domain_lists, grid, exinfo.block(bi, bj));
            }
        }
    }

    /**
     * @brief loops over all blocks and execute sequentially all mss functors for each block
     * @tparam MssComponents a meta array with the mss components of all MSS
     */
    template <class MssComponents,
        class LocalDomainListArray,
        class Grid,
        enable_if_t<_impl::all_mss_kparallel<MssComponents>::value, int> = 0>
    void fused_mss_loop(backend::mc, LocalDomainListArray const &local_domain_lists, const Grid &grid) {
        GT_STATIC_ASSERT((meta::all_of<is_mss_components, MssComponents>::value), GT_INTERNAL_ERROR);

        execinfo_mc exinfo(grid);
        const int_t i_blocks = exinfo.i_blocks();
        const int_t j_blocks = exinfo.j_blocks();
        const int_t k_first = grid.k_min();
        const int_t k_last = grid.k_max();
#pragma omp parallel for collapse(3)
        for (int_t bj = 0; bj < j_blocks; ++bj) {
            for (int_t k = k_first; k <= k_last; ++k) {
                for (int_t bi = 0; bi < i_blocks; ++bi) {
                    run_mss_functors<MssComponents>(backend::mc{}, local_domain_lists, grid, exinfo.block(bi, bj, k));
                }
            }
        }
    }

    /**
     * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
     */
    constexpr std::true_type mss_fuse_esfs(backend::mc) { return {}; }
} // namespace gridtools
