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

#include <utility>

#include "../../common/functional.hpp"
#include "../../common/timer/timer_traits.hpp"
#include "../backend_traits_fwd.hpp"
#include "../mss_functor.hpp"
#include "../structured_grids/backend_mc/execute_kernel_functor_mc.hpp"

/**@file
@brief type definitions and structures specific for the Mic backend
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

    /**Traits struct, containing the types which are specific for the mc backend*/
    template <>
    struct backend_traits<target::mc> {
        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         */
        template <typename RunFunctorArgs>
        struct mss_loop {
            typedef typename RunFunctorArgs::backend_target_t backend_target_t;

            GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
            template <typename LocalDomain, typename Grid, typename ExecutionInfo>
            GT_FUNCTION static void run(
                LocalDomain const &local_domain, Grid const &grid, const ExecutionInfo &execution_info) {
                GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                execute_kernel_functor_mc<RunFunctorArgs>(local_domain, grid)(execution_info);
            }
        };

        /**
         * @brief Loops over all blocks and executes sequentially all MSS functors for each block.
         * Implementation for stencils with serial execution along k-axis.
         *
         * @tparam MssComponents A meta array with the MSS components of all MSS.
         * @tparam BackendTarget IDs of backend.
         */
        template <typename MssComponents, typename BackendTarget, typename Enable = void>
        struct fused_mss_loop {
            GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);

            template <typename LocalDomainListArray, typename Grid>
            GT_FUNCTION static void run(LocalDomainListArray const &local_domain_lists, Grid const &grid) {
                using iter_range = GT_META_CALL(meta::make_indices, boost::mpl::size<MssComponents>);
                using mss_functor_t =
                    mss_functor<MssComponents, Grid, LocalDomainListArray, BackendTarget, execinfo_mc::block_kserial_t>;

                execinfo_mc exinfo(grid);
                const int_t i_blocks = exinfo.i_blocks();
                const int_t j_blocks = exinfo.j_blocks();
#pragma omp parallel for collapse(2)
                for (int_t bj = 0; bj < j_blocks; ++bj) {
                    for (int_t bi = 0; bi < i_blocks; ++bi) {
                        gridtools::for_each<iter_range>(mss_functor_t(local_domain_lists, grid, exinfo.block(bi, bj)));
                    }
                }
            }
        };

        /**
         * @brief Loops over all blocks and executes sequentially all MSS functors for each block.
         * Implementation for stencils with parallel execution along k-axis.
         *
         * @tparam MssComponents A meta array with the MSS components of all MSS.
         * @tparam BackendTarget ID of backend.
         */
        template <typename MssComponents, typename BackendTarget>
        struct fused_mss_loop<MssComponents,
            BackendTarget,
            typename std::enable_if<_impl::all_mss_kparallel<MssComponents>::value>::type> {
            GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);

            template <typename LocalDomainListArray, typename Grid>
            GT_FUNCTION static void run(LocalDomainListArray const &local_domain_lists, Grid const &grid) {
                using iter_range = GT_META_CALL(meta::make_indices, boost::mpl::size<MssComponents>);
                using mss_functor_t = mss_functor<MssComponents,
                    Grid,
                    LocalDomainListArray,
                    BackendTarget,
                    execinfo_mc::block_kparallel_t>;

                execinfo_mc exinfo(grid);
                const int_t i_blocks = exinfo.i_blocks();
                const int_t j_blocks = exinfo.j_blocks();
                const int_t k_first = grid.k_min();
                const int_t k_last = grid.k_max();
#pragma omp parallel for collapse(3)
                for (int_t bj = 0; bj < j_blocks; ++bj) {
                    for (int_t k = k_first; k <= k_last; ++k) {
                        for (int_t bi = 0; bi < i_blocks; ++bi) {
                            gridtools::for_each<iter_range>(
                                mss_functor_t(local_domain_lists, grid, exinfo.block(bi, bj, k)));
                        }
                    }
                }
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        using mss_fuse_esfs_strategy = std::true_type;

        using performance_meter_t = typename timer_traits<target::mc>::timer_type;
    };
} // namespace gridtools
