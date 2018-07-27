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

#include "../../../common/defs.hpp"
#include "../../../common/generic_metafunctions/for_each.hpp"
#include "../../../common/generic_metafunctions/is_sequence_of.hpp"
#include "../../../common/generic_metafunctions/meta.hpp"
#include "../../block.hpp"
#include "../../grid.hpp"
#include "../../mss_components.hpp"
#include "../../mss_functor.hpp"
#include "../../reductions/reduction_data.hpp"
#include "execute_kernel_functor_mic.hpp"

namespace gridtools {

    template <class>
    struct strategy_from_id_mic;

    /**
     * @brief struct holding backend-specific runtime information about stencil execution.
     */
    struct execution_info_mic {
        uint_t bi, bj;
    };

    /**
       @brief specialization for the \ref gridtools::strategy::block strategy
       The loops over i and j are split according to the values of BI and BJ
    */
    template <>
    struct strategy_from_id_mic<strategy::block> {
        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponents a meta array with the mss components of all MSS
         * @tparam BackendIds ids of backend
         */
        template <typename MssComponents, typename BackendIds, typename ReductionData>
        struct fused_mss_loop {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of<MssComponents, is_mss_components>::value), GT_INTERNAL_ERROR);

            using iter_range = GT_META_CALL(meta::make_indices, boost::mpl::size<MssComponents>);

            template <typename LocalDomainListArray, typename Grid>
            static void run(
                LocalDomainListArray const &local_domain_lists, Grid const &grid, ReductionData &reduction_data) {
                GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);

                const uint_t n = grid.i_high_bound() - grid.i_low_bound();
                const uint_t m = grid.j_high_bound() - grid.j_low_bound();

                const uint_t NBI = n / block_i_size(BackendIds{});
                const uint_t NBJ = m / block_j_size(BackendIds{});

#pragma omp parallel for
                for (uint_t bi = 0; bi <= NBI; ++bi) {
                    for (uint_t bj = 0; bj <= NBJ; ++bj) {
                        gridtools::for_each<iter_range>(mss_functor<MssComponents,
                            Grid,
                            LocalDomainListArray,
                            BackendIds,
                            ReductionData,
                            execution_info_mic>(local_domain_lists, grid, reduction_data, {bi, bj}));
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

            template <typename LocalDomain, typename Grid, typename ReductionData>
            static void run(const LocalDomain &local_domain,
                const Grid &grid,
                ReductionData &reduction_data,
                const execution_info_mic &execution_info) {
                GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT(
                    (boost::mpl::size<typename RunFunctorArgs::functor_list_t>::value == 1), GT_INTERNAL_ERROR);

                const uint_t n = grid.i_high_bound() - grid.i_low_bound();
                const uint_t m = grid.j_high_bound() - grid.j_low_bound();

                static constexpr auto BI = block_i_size(typename RunFunctorArgs::backend_ids_t{});
                static constexpr auto BJ = block_j_size(typename RunFunctorArgs::backend_ids_t{});

                const uint_t NBI = n / BI;
                const uint_t NBJ = m / BJ;

                const uint_t first_i = execution_info.bi * BI + grid.i_low_bound();
                const uint_t first_j = execution_info.bj * BJ + grid.j_low_bound();

                const uint_t last_i = execution_info.bi == NBI ? n - NBI * BI : BI - 1;
                const uint_t last_j = execution_info.bj == NBJ ? m - NBJ * BJ : BJ - 1;

                icgrid::execute_kernel_functor_mic<RunFunctorArgs>(local_domain,
                    grid,
                    reduction_data,
                    first_i,
                    first_j,
                    last_i,
                    last_j,
                    execution_info.bi,
                    execution_info.bj)();
            }
        };
    };

} // namespace gridtools
