/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#pragma once
#include "../backend_traits_fwd.hpp"
#include "../mss_functor.hpp"
#include "stencil-composition/backend_host/execute_kernel_functor_host.hpp"
#include "../../storage/meta_storage.hpp"
#include "../tile.hpp"

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
                (is_meta_array_of< MssComponentsArray, is_mss_components >::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), "Error");
            GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), "Error");

            typedef boost::mpl::range_c< uint_t,
                0,
                boost::mpl::size< typename MssComponentsArray::elements >::type::value > iter_range;

            template < typename LocalDomainListArray, typename Grid >
            static void run(LocalDomainListArray &local_domain_lists, const Grid &grid, ReductionData &reduction_data) {
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "Internal Error: wrong type");

                boost::mpl::for_each< iter_range >(
                    mss_functor< MssComponentsArray, Grid, LocalDomainListArray, BackendIds, ReductionData >(
                        local_domain_lists, grid, reduction_data, 0, 0));
            }
        };

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         */
        template < typename RunFunctorArgs >
        struct mss_loop {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArgs >::value), "Internal Error: wrong type");
            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;
            template < typename LocalDomain, typename Grid, typename ReductionData >
            static void run(const LocalDomain &local_domain,
                const Grid &grid,
                ReductionData &reduction_data,
                const uint_t bi,
                const uint_t bj) {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), "Internal Error: wrong type");
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "Internal Error: wrong type");
                GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), "Error");

                typedef grid_traits_from_id< backend_ids_t::s_grid_type_id > grid_traits_t;
                typedef
                    typename grid_traits_t::template with_arch< backend_ids_t::s_backend_id >::type arch_grid_traits_t;

                // getting the architecture and grid dependent traits
                typedef typename arch_grid_traits_t::template kernel_functor_executor< RunFunctorArgs >::type
                    kernel_functor_executor_t;

                typedef typename RunFunctorArgs::functor_list_t functor_list_t;
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< functor_list_t >::value == 1), "Internal Error: wrong size");
                kernel_functor_executor_t(local_domain, grid, reduction_data)();
            }
        };

        // NOTE: this part is (and should remain) an exact copy-paste in the naive, block, host and cuda versions
        template < typename Index,
            typename Layout,
            typename Halo,
            typename Alignment,
#ifdef CXX11_ENABLED
            typename... Tiles
#else
            typename TileI,
            typename TileJ
#endif
            >
        struct get_tmp_storage_info {
            GRIDTOOLS_STATIC_ASSERT(is_aligned< Alignment >::type::value, "wrong type");

            GRIDTOOLS_STATIC_ASSERT(is_layout_map< Layout >::value, "wrong type for layout map");
#ifdef CXX11_ENABLED
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_tile< Tiles >::type::value...), "wrong type for the tiles");
#else
            GRIDTOOLS_STATIC_ASSERT((is_tile< TileI >::value && is_tile< TileJ >::value), "wrong type for the tiles");
#endif
            GRIDTOOLS_STATIC_ASSERT(is_halo< Halo >::type::value, "wrong type");

            typedef meta_storage< meta_storage_aligned< meta_storage_base< Index, Layout, true >, Alignment, Halo > > type;
        };

/**
 * @brief metafunction that returns the storage type for the storage type of the temporaries for this strategy.
 */
// NOTE: this part is (and should remain) an exact copy-paste in the naive, block, host and cuda versions
#ifdef CXX11_ENABLED
        template < typename Storage, typename... Tiles >
#else
        template < typename Storage, typename TileI, typename TileJ >
#endif
        struct get_tmp_storage {
#ifdef CXX11_ENABLED
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_tile< Tiles >::type::value...), "wrong type for the tiles");
#else
            GRIDTOOLS_STATIC_ASSERT((is_tile< TileI >::value && is_tile< TileJ >::value), "wrong type for the tiles");
#endif
            typedef storage<
#ifdef CXX11_ENABLED
                typename Storage::template type_tt
#else
                base_storage
#endif
                < typename Storage::pointer_type,
                    typename get_tmp_storage_info< typename Storage::storage_info_type::index_type,
                        typename Storage::storage_info_type::layout,
                        typename Storage::storage_info_type::halo_t,
                        typename Storage::storage_info_type::alignment_t,
#ifdef CXX11_ENABLED
                        Tiles...
#else
                        TileI,
                        TileJ
#endif
                        >::type,
                    Storage::field_dimensions > > type;
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
                (is_meta_array_of< MssComponentsArray, is_mss_components >::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), "Error");
            GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), "Error");

            typedef boost::mpl::range_c< uint_t,
                0,
                boost::mpl::size< typename MssComponentsArray::elements >::type::value > iter_range;

            template < typename LocalDomainListArray, typename Grid >
            static void run(LocalDomainListArray &local_domain_lists, const Grid &grid, ReductionData &reduction_data) {
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "Internal Error: wrong type");

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
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments< RunFunctorArgs >::value), "Internal Error: wrong type");

            typedef typename RunFunctorArgs::backend_ids_t backend_ids_t;

            template < typename LocalDomain, typename Grid, typename ReductionData >
            static void run(const LocalDomain &local_domain,
                const Grid &grid,
                ReductionData &reduction_data,
                const uint_t bi,
                const uint_t bj) {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), "Internal Error: wrong type");
                GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "Internal Error: wrong type");
                GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), "Error");

                typedef grid_traits_from_id< backend_ids_t::s_grid_type_id > grid_traits_t;
                typedef
                    typename grid_traits_t::template with_arch< backend_ids_t::s_backend_id >::type arch_grid_traits_t;

                typedef typename arch_grid_traits_t::template kernel_functor_executor< RunFunctorArgs >::type
                    kernel_functor_executor_t;

                typedef typename RunFunctorArgs::functor_list_t functor_list_t;
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size< functor_list_t >::value == 1), "Internal Error: wrong size");

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

        // NOTE: this part is (and should remain) an exact copy-paste in the naive, block, host and cuda versions
        template < typename Index,
            typename Layout,
            typename Halo
#ifdef CXX11_ENABLED
            ,
            typename... Tiles
#else
            ,
            typename TileI,
            typename TileJ
#endif
            >
        struct get_tmp_meta_storage {
            GRIDTOOLS_STATIC_ASSERT(is_layout_map< Layout >::value, "wrong type for layout map");
#ifdef CXX11_ENABLED
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_tile< Tiles >::type::value...), "wrong type for the tiles");
#else
            GRIDTOOLS_STATIC_ASSERT((is_tile< TileI >::value && is_tile< TileJ >::value), "wrong type for the tiles");
#endif
            GRIDTOOLS_STATIC_ASSERT(is_halo< Halo >::type::value, "wrong type");

            typedef meta_storage_tmp<
                meta_storage_aligned< meta_storage_base< Index, Layout, true >, aligned< 0 >, Halo >
#ifdef CXX11_ENABLED
                ,
                Tiles...
#else
                ,
                TileI,
                TileJ
#endif
                > type;
        };

/**
 * @brief metafunction that returns the storage type for the storage type of the temporaries for this strategy.
 */
// NOTE: this part is (and should remain) an exact copy-paste in the naive, block, host and cuda versions
#ifdef CXX11_ENABLED
        template < typename Storage, typename... Tiles >
#else
        template < typename Storage, typename TileI, typename TileJ >
#endif
        struct get_tmp_storage {
#ifdef CXX11_ENABLED
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_tile< Tiles >::type::value...), "wrong type for the tiles");
#else
            GRIDTOOLS_STATIC_ASSERT((is_tile< TileI >::value && is_tile< TileJ >::value), "wrong type for the tiles");
#endif
            typedef storage<
#ifdef CXX11_ENABLED
                typename Storage::template type_tt
#else
                base_storage
#endif
                < typename Storage::pointer_type,
                    typename get_tmp_meta_storage< typename Storage::storage_info_type::index_type,
                        typename Storage::storage_info_type::layout,
                        typename Storage::storage_info_type::halo_t,
#ifdef CXX11_ENABLED
                        Tiles...
#else
                        TileI,
                        TileJ
#endif
                        >::type,
                    Storage::field_dimensions > > type;
        };
    };

} // namespace gridtools
