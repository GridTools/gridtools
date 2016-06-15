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

#include <gridtools.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/mpl/has_key.hpp>
#include "../level.hpp"

#include "../mss_functor.hpp"
#include "../sfinae.hpp"
#include "../../storage/meta_storage.hpp"
#include "../tile.hpp"
#include "common/generic_metafunctions/is_variadic_pack_of.hpp"
#include "execute_kernel_functor_cuda.hpp"

namespace gridtools {

    template < enumtype::strategy >
    struct strategy_from_id_cuda;

    /**
       @brief specialization for the \ref gridtools::_impl::Naive strategy
    */
    template <>
    struct strategy_from_id_cuda< enumtype::Naive > {};

    /**
       @brief specialization for the \ref gridtools::_impl::Block strategy
       Empty as not used in the CUDA backend
    */
    template <>
    struct strategy_from_id_cuda< enumtype::Block > {
        // default block size for Block strategy
        typedef block_size< GT_DEFAULT_TILE_I, GT_DEFAULT_TILE_J, 1 > block_size_t;

        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponentsArray a meta array with the mss components of all MSS
         * @tparam BackendIds backend ids type
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

        // NOTE: this part is (and should remain) an exact copy-paste in the naive, block, host and cuda versions
        template < typename Index,
            typename Layout,
            typename Halo,
            typename Alignment
#ifdef CXX11_ENABLED
            ,
            typename... Tiles
#else
            ,
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

            typedef meta_storage<
                meta_storage_tmp< meta_storage_aligned< meta_storage_base< Index::value, Layout, true >,
                                      Alignment, // alignment boundary
                                      Halo >,
#ifdef CXX11_ENABLED
                    Tiles...
#else
                    TileI,
                    TileJ
#endif
                    > > type;
        };

/**
 * @brief metafunction that returns the storage type for the storage type of the temporaries for this strategy.
 * with the naive algorithms, the temporary storages are like the non temporary ones
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

} // namespace gridtools
