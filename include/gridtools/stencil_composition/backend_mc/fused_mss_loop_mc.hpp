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

#include <boost/mpl/size.hpp>

#include "../mss_functor.hpp"

/**@file
 * @brief fused mss loop implementations for the mc backend
 */
namespace gridtools {
    namespace _impl {
        /**
         * @brief Meta function to check if an MSS can be executed in parallel along k-axis.
         */
        struct is_mss_kparallel {
            template <typename Mss>
            struct apply {
                using type = execute::is_parallel<typename Mss::execution_engine_t>;
            };
        };

        /**
         * @brief Meta function to check if all MSS in an MssComponents array can be executed in parallel along k-axis.
         */
        template <typename MssComponents>
        struct all_mss_kparallel
            : boost::mpl::fold<typename boost::mpl::transform<MssComponents, is_mss_kparallel>::type,
                  boost::mpl::true_,
                  boost::mpl::and_<boost::mpl::placeholders::_1, boost::mpl::placeholders::_2>>::type {};
    } // namespace _impl

    /**
     * @brief loops over all blocks and execute sequentially all mss functors for each block
     * @tparam MssComponents a meta array with the mss components of all MSS
     */
    template <class MssComponents,
        class LocalDomainListArray,
        class Grid,
        enable_if_t<!_impl::all_mss_kparallel<MssComponents>::value, int> = 0>
    static void fused_mss_loop(
        target::mc const &backend_target, LocalDomainListArray const &local_domain_lists, const Grid &grid) {
        GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);

        execinfo_mc exinfo(grid);
        const int_t i_blocks = exinfo.i_blocks();
        const int_t j_blocks = exinfo.j_blocks();
#pragma omp parallel for collapse(2)
        for (int_t bj = 0; bj < j_blocks; ++bj) {
            for (int_t bi = 0; bi < i_blocks; ++bi) {
                host::for_each<GT_META_CALL(meta::make_indices, boost::mpl::size<MssComponents>)>(
                    make_mss_functor<MssComponents>(backend_target, local_domain_lists, grid, exinfo.block(bi, bj)));
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
    static void fused_mss_loop(
        target::mc const &backend_target, LocalDomainListArray const &local_domain_lists, const Grid &grid) {
        GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);

        execinfo_mc exinfo(grid);
        const int_t i_blocks = exinfo.i_blocks();
        const int_t j_blocks = exinfo.j_blocks();
        const int_t k_first = grid.k_min();
        const int_t k_last = grid.k_max();
#pragma omp parallel for collapse(3)
        for (int_t bj = 0; bj < j_blocks; ++bj) {
            for (int_t k = k_first; k <= k_last; ++k) {
                for (int_t bi = 0; bi < i_blocks; ++bi) {
                    host::for_each<GT_META_CALL(meta::make_indices, boost::mpl::size<MssComponents>)>(
                        make_mss_functor<MssComponents>(
                            backend_target, local_domain_lists, grid, exinfo.block(bi, bj, k)));
                }
            }
        }
    }
} // namespace gridtools
