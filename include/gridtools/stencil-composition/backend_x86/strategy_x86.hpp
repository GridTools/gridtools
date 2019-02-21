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

#include <boost/mpl/for_each.hpp>

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/is_sequence_of.hpp"
#include "../backend_ids.hpp"
#include "../block.hpp"
#include "../grid.hpp"
#include "../mss_components.hpp"
#include "../mss_functor.hpp"
#include "./execute_kernel_functor_x86.hpp"

namespace gridtools {

    template <class>
    struct strategy_from_id_x86;

    /**
     * @brief struct holding backend-specific runtime information about stencil execution.
     */
    struct execution_info_x86 {
        uint_t bi, bj;
    };

    /**
       @brief specialization for the \ref strategy::naive strategy
    */
    template <>
    struct strategy_from_id_x86<strategy::naive> {
        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponents a meta array with the mss components of all MSS
         * @tparam BackendIds ids of backend
         */
        template <typename MssComponents, typename BackendIds>
        struct fused_mss_loop {
            GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);

            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<MssComponents>::type::value> iter_range;

            template <typename LocalDomainListArray, typename Grid>
            static void run(LocalDomainListArray const &local_domain_lists, const Grid &grid) {
                GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                boost::mpl::for_each<iter_range>(
                    mss_functor<MssComponents, Grid, LocalDomainListArray, BackendIds, execution_info_x86>{
                        local_domain_lists, grid, {0, 0}});
            }
        };

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         */
        template <typename RunFunctorArgs>
        struct mss_loop {
            GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;
            template <typename LocalDomain, typename Grid>
            static void run(
                const LocalDomain &local_domain, const Grid &grid, const execution_info_x86 &execution_info) {
                GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                // getting the architecture and grid dependent traits
                typedef typename kernel_functor_executor<backend_ids_t, RunFunctorArgs>::type kernel_functor_executor_t;

                kernel_functor_executor_t{local_domain,
                    grid,
                    grid.i_high_bound() - grid.i_low_bound() + 1,
                    grid.j_high_bound() - grid.j_low_bound() + 1,
                    0,
                    0}();
            }
        };
    };

    /**
       @brief specialization for the \ref strategy::block strategy
       The loops over i and j are split according to the values of BI and BJ
    */
    template <>
    struct strategy_from_id_x86<strategy::block> {
        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponents a meta array with the mss components of all MSS
         * @tparam BackendIds ids of backend
         */
        template <typename MssComponents, typename BackendIds>
        struct fused_mss_loop {
            GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);
            GT_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);

            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<MssComponents>::type::value> iter_range;

            template <typename LocalDomainListArray, typename Grid>
            static void run(LocalDomainListArray const &local_domain_lists, const Grid &grid) {
                GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                uint_t n = grid.i_high_bound() - grid.i_low_bound();
                uint_t m = grid.j_high_bound() - grid.j_low_bound();

                uint_t NBI = n / block_i_size(BackendIds{});
                uint_t NBJ = m / block_j_size(BackendIds{});

#pragma omp parallel
                {
#pragma omp for nowait
                    for (uint_t bi = 0; bi <= NBI; ++bi) {
                        for (uint_t bj = 0; bj <= NBJ; ++bj) {
                            boost::mpl::for_each<iter_range>(
                                mss_functor<MssComponents, Grid, LocalDomainListArray, BackendIds, execution_info_x86>(
                                    local_domain_lists, grid, {bi, bj}));
                        }
                    }
                }
            }
        };

        /**
         * @brief main execution of a mss for a given IJ block. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         */
        template <typename RunFunctorArgs>
        struct mss_loop {
            GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);

            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;

            template <typename LocalDomain, typename Grid>
            static void run(
                const LocalDomain &local_domain, const Grid &grid, const execution_info_x86 &execution_info) {
                GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                typedef typename kernel_functor_executor<backend_ids_t, RunFunctorArgs>::type kernel_functor_executor_t;

                auto block_size_f = [](uint_t total, uint_t block_size, uint_t block_no) {
                    auto n = (total + block_size - 1) / block_size;
                    return block_no == n - 1 ? total - block_no * block_size : block_size;
                };
                auto total_i = grid.i_high_bound() - grid.i_low_bound() + 1;
                auto total_j = grid.j_high_bound() - grid.j_low_bound() + 1;

                kernel_functor_executor_t{local_domain,
                    grid,
                    block_size_f(total_i, block_i_size(backend_ids_t{}), execution_info.bi),
                    block_size_f(total_j, block_j_size(backend_ids_t{}), execution_info.bj),
                    execution_info.bi,
                    execution_info.bj}();
            }
        };
    };

} // namespace gridtools
