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

#include "../backend_traits_fwd.hpp"
#include "../mss_functor.hpp"
#include "../tile.hpp"
#include "execute_kernel_functor_mic.hpp"

namespace gridtools {

    template < enumtype::strategy >
    struct strategy_from_id_mic;

    template <>
    struct strategy_from_id_mic< enumtype::Naive > {};

    /**
       @brief specialization for the \ref gridtools::_impl::Block strategy
       The loops over i and j are split according to the values of BI and BJ
    */
    template <>
    struct strategy_from_id_mic< enumtype::Block > {
        using block_size_t = block_size< 0, 0, 0 >;

        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponentsArray a meta array with the mss components of all MSS
         * @tparam BackendIds ids of backend
         */
        template < typename MssComponentsArray, typename BackendIds, typename ReductionData >
        struct fused_mss_loop {
            GRIDTOOLS_STATIC_ASSERT(
                (is_meta_array_of< MssComponentsArray, is_mss_components >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), GT_INTERNAL_ERROR);
            GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), GT_INTERNAL_ERROR);

          private:
            // meta function to check if an MssComponent can be executed in parallel along k-axis
            template < typename MssComponents >
            struct is_parallel {
                using type =
                    boost::mpl::bool_< MssComponents::execution_engine_t::type::execution == enumtype::parallel_impl >;
            };

            // boolean type that is true iff all MssComponents can be executed in parallel along k-axis
            using all_parallel =
                typename boost::mpl::fold< typename boost::mpl::transform< typename MssComponentsArray::elements,
                                               is_parallel< boost::mpl::placeholders::_1 > >::type,
                    boost::mpl::true_,
                    boost::mpl::and_< boost::mpl::placeholders::_1, boost::mpl::placeholders::_2 > >::type;

            using iter_range = boost::mpl::range_c< uint_t,
                0,
                boost::mpl::size< typename MssComponentsArray::elements >::type::value >;

          public:
            template < typename LocalDomainListArray, typename Grid >
            static void run(LocalDomainListArray &local_domain_lists, const Grid &grid, ReductionData &reduction_data) {
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);

                run(local_domain_lists, grid, reduction_data, all_parallel());
            }

          private:
            template < typename LocalDomainListArray, typename Grid >
            static void run(LocalDomainListArray &local_domain_lists,
                const Grid &grid,
                ReductionData &reduction_data,
                boost::mpl::true_) {
                using mss_functor_t = mss_functor< MssComponentsArray,
                    Grid,
                    LocalDomainListArray,
                    BackendIds,
                    ReductionData,
                    execution_info_parallel_mic >;

                int_t i_block_size, j_block_size;
                std::tie(i_block_size, j_block_size) = block_size_mic(grid);

                const int_t i_grid_size = grid.i_high_bound() - grid.i_low_bound() + 1;
                const int_t j_grid_size = grid.j_high_bound() - grid.j_low_bound() + 1;

                const int_t i_blocks = (i_grid_size + i_block_size - 1) / i_block_size;
                const int_t j_blocks = (j_grid_size + j_block_size - 1) / j_block_size;

                const int_t k_first = grid.k_min();
                const int_t k_last = grid.k_max();
#pragma omp parallel for collapse(3)
                for (int_t k = k_first; k <= k_last; ++k) {
                    for (int_t bj = 0; bj < j_blocks; ++bj) {
                        for (int_t bi = 0; bi < i_blocks; ++bi) {
                            boost::mpl::for_each< iter_range >(
                                mss_functor_t(local_domain_lists, grid, reduction_data, {bi, bj, k}));
                        }
                    }
                }
                reduction_data.reduce();
            }

            template < typename LocalDomainListArray, typename Grid >
            static void run(LocalDomainListArray &local_domain_lists,
                const Grid &grid,
                ReductionData &reduction_data,
                boost::mpl::false_) {
                using mss_functor_t = mss_functor< MssComponentsArray,
                    Grid,
                    LocalDomainListArray,
                    BackendIds,
                    ReductionData,
                    execution_info_serial_mic >;

                int_t i_block_size, j_block_size;
                std::tie(i_block_size, j_block_size) = block_size_mic(grid);

                const int_t i_grid_size = grid.i_high_bound() - grid.i_low_bound() + 1;
                const int_t j_grid_size = grid.j_high_bound() - grid.j_low_bound() + 1;

                const int_t i_blocks = (i_grid_size + i_block_size - 1) / i_block_size;
                const int_t j_blocks = (j_grid_size + j_block_size - 1) / j_block_size;

                const int_t k_first = grid.k_min();
                const int_t k_last = grid.k_max();
#pragma omp parallel for collapse(2)
                for (int_t bj = 0; bj < j_blocks; ++bj) {
                    for (int_t bi = 0; bi < i_blocks; ++bi) {
                        boost::mpl::for_each< iter_range >(
                            mss_functor_t(local_domain_lists, grid, reduction_data, {bi, bj}));
                    }
                }
                reduction_data.reduce();
            }
        };
    };

} // namespace gridtools
