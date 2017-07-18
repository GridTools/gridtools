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
#include "../backend_traits_fwd.hpp"
#include "../mss_functor.hpp"
#include "../mss_functor_serializable.hpp"
#include "../tile.hpp"
#include "../../common/stencil_serializer.hpp"
#include "execute_kernel_functor_host.hpp"

namespace gridtools {

    template < enumtype::strategy >
    struct strategy_from_id_host;

    /**
       @brief specialization for the \ref gridtools::_impl::Naive strategy
    */
    template <>
    struct strategy_from_id_host< enumtype::Naive > {
        // default block size for Naive strategy
        typedef block_size< 0, 0, 0 > block_size_t;
        static const uint_t BI = block_size_t::i_size_t::value;
        static const uint_t BJ = block_size_t::j_size_t::value;
        static const uint_t BK = 0;

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

            typedef boost::mpl::range_c< uint_t,
                0,
                boost::mpl::size< typename MssComponentsArray::elements >::type::value > iter_range;

            template < typename LocalDomainListArray, typename Grid >
            static void run(LocalDomainListArray &local_domain_lists, const Grid &grid, ReductionData &reduction_data) {
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);

                boost::mpl::for_each< iter_range >(
                    mss_functor< MssComponentsArray, Grid, LocalDomainListArray, BackendIds, ReductionData >(
                        local_domain_lists, grid, reduction_data, 0, 0));
            }

            template < typename LocalDomainListArray, typename Domain, typename Grid, class SerializerType >
            static void run_and_serialize(LocalDomainListArray &local_domain_lists,
                const Domain &domain,
                const Grid &grid,
                ReductionData &reduction_data,
                stencil_serializer< SerializerType > &stencil_ser) {
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_aggregator_type< Domain >::value), GT_INTERNAL_ERROR);

                boost::mpl::for_each< iter_range >(mss_functor_serializable< MssComponentsArray,
                    Domain,
                    Grid,
                    LocalDomainListArray,
                    BackendIds,
                    ReductionData,
                    SerializerType >(local_domain_lists, domain, grid, reduction_data, 0, 0, stencil_ser));
            }
        };

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         */
        template < typename RunFunctorArgs >
        struct mss_loop {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArgs >::value), GT_INTERNAL_ERROR);
            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;
            template < typename LocalDomain, typename Grid, typename ReductionData >
            static void run(const LocalDomain &local_domain,
                const Grid &grid,
                ReductionData &reduction_data,
                const uint_t bi,
                const uint_t bj) {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), GT_INTERNAL_ERROR);

                typedef grid_traits_from_id< backend_ids_t::s_grid_type_id > grid_traits_t;
                typedef
                    typename grid_traits_t::template with_arch< backend_ids_t::s_backend_id >::type arch_grid_traits_t;

                // getting the architecture and grid dependent traits
                typedef typename arch_grid_traits_t::template kernel_functor_executor< RunFunctorArgs >::type
                    kernel_functor_executor_t;

                typedef typename RunFunctorArgs::functor_list_t functor_list_t;
                GRIDTOOLS_STATIC_ASSERT(
                    (boost::mpl::size< functor_list_t >::value == 1), GT_INTERNAL_ERROR_MSG("Wrong Size"));
                kernel_functor_executor_t(local_domain, grid, reduction_data)();
            }
        };
    };

    /**
       @brief specialization for the \ref gridtools::_impl::Block strategy
       The loops over i and j are split according to the values of BI and BJ
    */
    template <>
    struct strategy_from_id_host< enumtype::Block > {
        // default block size for Block strategy
        typedef block_size< GT_DEFAULT_TILE_I, GT_DEFAULT_TILE_J, 1 > block_size_t;

        static const uint_t BI = block_size_t::i_size_t::value;
        static const uint_t BJ = block_size_t::j_size_t::value;
        static const uint_t BK = 0;

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

            typedef boost::mpl::range_c< uint_t,
                0,
                boost::mpl::size< typename MssComponentsArray::elements >::type::value > iter_range;

            template < typename LocalDomainListArray, typename Grid >
            static void run(LocalDomainListArray &local_domain_lists, const Grid &grid, ReductionData &reduction_data) {
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);

                uint_t n = grid.i_high_bound() - grid.i_low_bound();
                uint_t m = grid.j_high_bound() - grid.j_low_bound();

                uint_t NBI = n / BI;
                uint_t NBJ = m / BJ;

#pragma omp parallel
                {
#pragma omp for nowait
                    for (uint_t bi = 0; bi <= NBI; ++bi) {
                        for (uint_t bj = 0; bj <= NBJ; ++bj) {
                            boost::mpl::for_each< iter_range >(mss_functor< MssComponentsArray,
                                Grid,
                                LocalDomainListArray,
                                BackendIds,
                                ReductionData >(local_domain_lists, grid, reduction_data, bi, bj));
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
        template < typename RunFunctorArgs >
        struct mss_loop {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArgs >::value), GT_INTERNAL_ERROR);

            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;

            template < typename LocalDomain, typename Grid, typename ReductionData >
            static void run(const LocalDomain &local_domain,
                const Grid &grid,
                ReductionData &reduction_data,
                const uint_t bi,
                const uint_t bj) {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), GT_INTERNAL_ERROR);
                GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), GT_INTERNAL_ERROR);

                typedef grid_traits_from_id< backend_ids_t::s_grid_type_id > grid_traits_t;
                typedef
                    typename grid_traits_t::template with_arch< backend_ids_t::s_backend_id >::type arch_grid_traits_t;

                typedef typename arch_grid_traits_t::template kernel_functor_executor< RunFunctorArgs >::type
                    kernel_functor_executor_t;

                typedef typename RunFunctorArgs::functor_list_t functor_list_t;
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< functor_list_t >::value == 1), GT_INTERNAL_ERROR);

                uint_t n = grid.i_high_bound() - grid.i_low_bound();
                uint_t m = grid.j_high_bound() - grid.j_low_bound();

                uint_t NBI = n / BI;
                uint_t NBJ = m / BJ;

                uint_t first_i = bi * BI + grid.i_low_bound();
                uint_t first_j = bj * BJ + grid.j_low_bound();

                uint_t last_i = BI - 1;
                uint_t last_j = BJ - 1;

                if (bi == NBI && bj == NBJ) {
                    last_i = n - NBI * BI;
                    last_j = m - NBJ * BJ;
                } else if (bi == NBI) {
                    last_i = n - NBI * BI;
                } else if (bj == NBJ) {
                    last_j = m - NBJ * BJ;
                }

                kernel_functor_executor_t(
                    local_domain, grid, reduction_data, first_i, first_j, last_i, last_j, bi, bj)();
            }
        };
    };

} // namespace gridtools
