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
#include <type_traits>

#include "../../backend_traits_fwd.hpp"
#include "../../mss_functor.hpp"
#include "../../tile.hpp"
#include "execute_kernel_functor_mic.hpp"
#include "execinfo_mic.hpp"
#include "common/generic_metafunctions/for_each.hpp"
#include "common/generic_metafunctions/meta.hpp"

namespace gridtools {

    template < enumtype::strategy >
    struct strategy_from_id_mic;

    template <>
    struct strategy_from_id_mic< enumtype::Naive > {};

    namespace _impl {

        /**
         * @brief Meta function to check if an MSS can be executed in parallel along k-axis.
         */
        struct is_mss_kparallel {
            template < typename Mss >
            struct apply {
                using type = boost::mpl::bool_< Mss::execution_engine_t::type::execution == enumtype::parallel_impl >;
            };
        };

        /**
         * @brief Meta function to check if all MSS in an MssComponents array can be executed in parallel along k-axis.
         */
        template < typename MssComponents >
        struct all_mss_kparallel
            : boost::mpl::fold< typename boost::mpl::transform< MssComponents, is_mss_kparallel >::type,
                  boost::mpl::true_,
                  boost::mpl::and_< boost::mpl::placeholders::_1, boost::mpl::placeholders::_2 > >::type {};

    } // namespace _impl

    /**
     * @brief Specialization for the \ref gridtools::_impl::Block strategy.
     */
    template <>
    struct strategy_from_id_mic< enumtype::Block > {
        using block_size_t = block_size< 0, 0, 0 >;

        /**
         * @brief Loops over all blocks and executes sequentially all MSS functors for each block.
         * Implementation for stencils with serial execution along k-axis.
         *
         * @tparam MssComponents A meta array with the MSS components of all MSS.
         * @tparam BackendIds IDs of backend.
         */
        template < typename MssComponents, typename BackendIds, typename ReductionData, typename Enable = void >
        struct fused_mss_loop {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< MssComponents, is_mss_components >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), GT_INTERNAL_ERROR);

            template < typename LocalDomainListArray, typename Grid >
            GT_FUNCTION static void run(
                LocalDomainListArray &local_domain_lists, const Grid &grid, ReductionData &reduction_data) {
                using iter_range = GT_META_CALL(meta::make_indices_for, MssComponents);
                using mss_functor_t = mss_functor< MssComponents,
                    Grid,
                    LocalDomainListArray,
                    BackendIds,
                    ReductionData,
                    execinfo_mic::block_kserial_t >;

                execinfo_mic exinfo(grid);
                const int_t i_blocks = exinfo.i_blocks();
                const int_t j_blocks = exinfo.j_blocks();
#pragma omp parallel for collapse(2)
                for (int_t bj = 0; bj < j_blocks; ++bj) {
                    for (int_t bi = 0; bi < i_blocks; ++bi) {
                        gridtools::for_each< iter_range >(
                            mss_functor_t(local_domain_lists, grid, reduction_data, exinfo.block(bi, bj)));
                    }
                }
                reduction_data.reduce();
            }
        };

        /**
         * @brief Loops over all blocks and executes sequentially all MSS functors for each block.
         * Implementation for stencils with parallel execution along k-axis.
         *
         * @tparam MssComponents A meta array with the MSS components of all MSS.
         * @tparam BackendIds IDs of backend.
         */
        template < typename MssComponents, typename BackendIds, typename ReductionData >
        struct fused_mss_loop< MssComponents,
            BackendIds,
            ReductionData,
            typename std::enable_if< _impl::all_mss_kparallel< MssComponents >::value >::type > {
            GRIDTOOLS_STATIC_ASSERT((is_sequence_of< MssComponents, is_mss_components >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), GT_INTERNAL_ERROR);

            template < typename LocalDomainListArray, typename Grid >
            GT_FUNCTION static void run(
                LocalDomainListArray &local_domain_lists, const Grid &grid, ReductionData &reduction_data) {
                using iter_range = GT_META_CALL(meta::make_indices_for, MssComponents);
                using mss_functor_t = mss_functor< MssComponents,
                    Grid,
                    LocalDomainListArray,
                    BackendIds,
                    ReductionData,
                    execinfo_mic::block_kparallel_t >;

                execinfo_mic exinfo(grid);
                const int_t i_blocks = exinfo.i_blocks();
                const int_t j_blocks = exinfo.j_blocks();
                const int_t k_first = grid.k_min();
                const int_t k_last = grid.k_max();
#pragma omp parallel for collapse(3)
                for (int_t bj = 0; bj < j_blocks; ++bj) {
                    for (int_t k = k_first; k <= k_last; ++k) {
                        for (int_t bi = 0; bi < i_blocks; ++bi) {
                            gridtools::for_each< iter_range >(
                                mss_functor_t(local_domain_lists, grid, reduction_data, exinfo.block(bi, bj, k)));
                        }
                    }
                }
                reduction_data.reduce();
            }
        };
    };

} // namespace gridtools
