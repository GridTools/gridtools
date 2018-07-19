/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
#pragma once

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/is_sequence_of.hpp"
#include "../backend_ids.hpp"
#include "../block.hpp"
#include "../grid.hpp"
#include "../mss_components.hpp"
#include "../mss_functor.hpp"
#include "../reductions/reduction_data.hpp"
#include "./execute_kernel_functor_host.hpp"

namespace gridtools {

    template <class>
    struct strategy_from_id_host;

    /**
     * @brief struct holding backend-specific runtime information about stencil execution.
     */
    struct execution_info_host {
        uint_t bi, bj;
    };

    /**
       @brief specialization for the \ref enumtype::strategy_naive strategy
    */
    template <>
    struct strategy_from_id_host<enumtype::strategy_naive> {
        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponents a meta array with the mss components of all MSS
         * @tparam BackendIds ids of backend
         */
        template <typename MssComponents, typename BackendIds, typename ReductionData>
        struct fused_mss_loop {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_reduction_data<ReductionData>::value), GT_INTERNAL_ERROR);

            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<MssComponents>::type::value> iter_range;

            template <typename LocalDomainListArray, typename Grid>
            static void run(
                LocalDomainListArray const &local_domain_lists, const Grid &grid, ReductionData &reduction_data) {
                GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                boost::mpl::for_each<iter_range>(mss_functor<MssComponents,
                    Grid,
                    LocalDomainListArray,
                    BackendIds,
                    ReductionData,
                    execution_info_host>{local_domain_lists, grid, reduction_data, {0, 0}});
            }
        };

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         */
        template <typename RunFunctorArgs>
        struct mss_loop {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);
            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;
            template <typename LocalDomain, typename Grid, typename ReductionData>
            static void run(const LocalDomain &local_domain,
                const Grid &grid,
                ReductionData &reduction_data,
                const execution_info_host &execution_info) {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_reduction_data<ReductionData>::value), GT_INTERNAL_ERROR);

                // getting the architecture and grid dependent traits
                typedef typename kernel_functor_executor<backend_ids_t, RunFunctorArgs>::type kernel_functor_executor_t;

                typedef typename RunFunctorArgs::functor_list_t functor_list_t;
                GRIDTOOLS_STATIC_ASSERT(
                    (boost::mpl::size<functor_list_t>::value == 1), GT_INTERNAL_ERROR_MSG("Wrong Size"));

                kernel_functor_executor_t{local_domain,
                    grid,
                    reduction_data,
                    grid.i_high_bound() - grid.i_low_bound() + 1,
                    grid.j_high_bound() - grid.j_low_bound() + 1,
                    0,
                    0}();
            }
        };
    };

    /**
       @brief specialization for the \ref enumtype::strategy_block strategy
       The loops over i and j are split according to the values of BI and BJ
    */
    template <>
    struct strategy_from_id_host<enumtype::strategy_block> {
        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponents a meta array with the mss components of all MSS
         * @tparam BackendIds ids of backend
         */
        template <typename MssComponents, typename BackendIds, typename ReductionData>
        struct fused_mss_loop {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_reduction_data<ReductionData>::value), GT_INTERNAL_ERROR);

            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<MssComponents>::type::value> iter_range;

            template <typename LocalDomainListArray, typename Grid>
            static void run(
                LocalDomainListArray const &local_domain_lists, const Grid &grid, ReductionData &reduction_data) {
                GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                uint_t n = grid.i_high_bound() - grid.i_low_bound();
                uint_t m = grid.j_high_bound() - grid.j_low_bound();

                uint_t NBI = n / block_i_size(BackendIds{});
                uint_t NBJ = m / block_j_size(BackendIds{});

#pragma omp parallel
                {
#pragma omp for nowait
                    for (uint_t bi = 0; bi <= NBI; ++bi) {
                        for (uint_t bj = 0; bj <= NBJ; ++bj) {
                            boost::mpl::for_each<iter_range>(mss_functor<MssComponents,
                                Grid,
                                LocalDomainListArray,
                                BackendIds,
                                ReductionData,
                                execution_info_host>(local_domain_lists, grid, reduction_data, {bi, bj}));
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
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), GT_INTERNAL_ERROR);

            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;

            template <typename LocalDomain, typename Grid, typename ReductionData>
            static void run(const LocalDomain &local_domain,
                const Grid &grid,
                ReductionData &reduction_data,
                const execution_info_host &execution_info) {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_reduction_data<ReductionData>::value), GT_INTERNAL_ERROR);

                typedef typename kernel_functor_executor<backend_ids_t, RunFunctorArgs>::type kernel_functor_executor_t;

                typedef typename RunFunctorArgs::functor_list_t functor_list_t;
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<functor_list_t>::value == 1), GT_INTERNAL_ERROR);

                auto block_size_f = [](uint_t total, uint_t block_size, uint_t block_no) {
                    auto n = (total + block_size - 1) / block_size;
                    return block_no == n - 1 ? total - block_no * block_size : block_size;
                };
                auto total_i = grid.i_high_bound() - grid.i_low_bound() + 1;
                auto total_j = grid.j_high_bound() - grid.j_low_bound() + 1;

                kernel_functor_executor_t{local_domain,
                    grid,
                    reduction_data,
                    block_size_f(total_i, block_i_size(backend_ids_t{}), execution_info.bi),
                    block_size_f(total_j, block_j_size(backend_ids_t{}), execution_info.bj),
                    execution_info.bi,
                    execution_info.bj}();
            }
        };
    };

} // namespace gridtools
