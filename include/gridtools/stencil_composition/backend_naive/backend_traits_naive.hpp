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

#include <boost/mpl/for_each.hpp>

#include "../../common/functional.hpp"
#include "../../common/timer/timer_traits.hpp"
#include "../backend_traits_fwd.hpp"
#include "../mss_functor.hpp"
#include "./execute_kernel_functor_naive.hpp"

/**@file
 * @brief type definitions and structures specific for the naive backend
 */
namespace gridtools {
    /**Traits struct, containing the types which are specific for the naive backend*/
    template <>
    struct backend_traits<target::naive> {
        /**
         * @brief struct holding backend-specific runtime information about stencil execution.
         */
        struct execution_info_naive {};

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         */
        template <typename RunFunctorArgs>
        struct mss_loop {
            GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);

            typedef typename RunFunctorArgs::backend_target_t backend_target_t;

            GT_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
            template <typename LocalDomain, typename Grid>
            static void run(
                LocalDomain const &local_domain, Grid const &grid, const execution_info_naive &execution_info) {
                GT_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
                GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                execute_kernel_functor_naive<RunFunctorArgs>{local_domain, grid}();
            }
        };
        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponents a meta array with the mss components of all MSS
         * @tparam BackendTarget ids of backend
         */
        template <typename MssComponents, typename BackendTarget>
        struct fused_mss_loop {
            GT_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);

            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<MssComponents>::type::value> iter_range;

            template <typename LocalDomainListArray, typename Grid>
            static void run(LocalDomainListArray const &local_domain_lists, const Grid &grid) {
                GT_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                boost::mpl::for_each<iter_range>(
                    mss_functor<MssComponents, Grid, LocalDomainListArray, BackendTarget, execution_info_naive>{
                        local_domain_lists, grid, {}});
            }
        };

        /**
         * @brief determines whether ESFs should be fused in one single kernel execution or not for this backend.
         */
        typedef std::false_type mss_fuse_esfs_strategy;

        using performance_meter_t = typename timer_traits<target::naive>::timer_type;
    };

} // namespace gridtools
