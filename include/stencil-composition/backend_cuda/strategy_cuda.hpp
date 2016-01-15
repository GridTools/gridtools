#pragma once

#include <gridtools.hpp>
#include <boost/fusion/include/value_at.hpp>
#include <boost/mpl/has_key.hpp>
#include "../level.hpp"

#include "backend_traits_cuda.hpp"
#include "../mss_functor.hpp"
#include "../sfinae.hpp"
#include "../../storage/meta_storage.hpp"
#include "../tile.hpp"
#include "common/generic_metafunctions/is_variadic_pack_of.hpp"

namespace gridtools{

    template<enumtype::strategy>
    struct strategy_from_id_cuda;

    /**
       @brief specialization for the \ref gridtools::_impl::Naive strategy
    */
    template<>
    struct strategy_from_id_cuda< enumtype::Naive>
    {
    };

    /**
       @brief specialization for the \ref gridtools::_impl::Block strategy
       Empty as not used in the CUDA backend
    */
    template<>
    struct strategy_from_id_cuda <enumtype::Block> {
        // default block size for Block strategy
        typedef block_size<GT_DEFAULT_TILE_I, GT_DEFAULT_TILE_J> block_size_t;

        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponentsArray a meta array with the mss components of all MSS
         * @tparam BackendId id of the backend
         */
        template<typename MssComponentsArray, enumtype::platform BackendId>
        struct fused_mss_loop
        {
            GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value), "Internal Error: wrong type");
            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename MssComponentsArray::elements>::type::value> iter_range;

            template<typename LocalDomainListArray, typename Grid>
            static void run(LocalDomainListArray& local_domain_lists, const Grid& grid)
            {
                GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), "Internal Error: wrong type");

                typedef backend_traits_from_id< BackendId > backend_traits;
                gridtools::for_each<iter_range> (
                    mss_functor<MssComponentsArray, Grid, LocalDomainListArray, BackendId, enumtype::Block>
                            (local_domain_lists, grid,0,0)
                );
            }
        };

        //NOTE: this part is (and should remain) an exact copy-paste in the naive, block, host and cuda versions
        template <typename Index, typename Layout, typename Halo, typename AlignmentBoundary
#ifdef CXX11_ENABLED
                  , typename ... Tiles
#else
                  , typename TileI, typename TileJ
#endif
                  >
        struct get_tmp_storage_info
        {
            GRIDTOOLS_STATIC_ASSERT(is_aligned<AlignmentBoundary>::type::value,"wrong type");
            GRIDTOOLS_STATIC_ASSERT(is_layout_map<Layout>::value, "wrong type for layout map");
#ifdef CXX11_ENABLED
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_tile<Tiles>::type::value ... ), "wrong type for the tiles");
#else
            GRIDTOOLS_STATIC_ASSERT((is_tile<TileI>::value && is_tile<TileJ>::value), "wrong type for the tiles");
#endif
            GRIDTOOLS_STATIC_ASSERT(is_halo<Halo>::type::value,"wrong type");

            typedef meta_storage
            <meta_storage_tmp
             <meta_storage_aligned
              <meta_storage_base
               <Index::value, Layout, true>,
               AlignmentBoundary ,//alignment boundary
               Halo
               >,
#ifdef CXX11_ENABLED
              Tiles ...
#else
              TileI, TileJ
#endif
              > > type;
        };

        /**
         * @brief metafunction that returns the storage type for the storage type of the temporaries for this strategy.
         * with the naive algorithms, the temporary storages are like the non temporary ones
         */
        //NOTE: this part is (and should remain) an exact copy-paste in the naive, block, host and cuda versions
#ifdef CXX11_ENABLED
        template <typename Storage, typename ... Tiles>
#else
        template <typename Storage, typename TileI, typename TileJ>
#endif
        struct get_tmp_storage
        {
#ifdef CXX11_ENABLED
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_tile<Tiles>::type::value ... ), "wrong type for the tiles");
#else
            GRIDTOOLS_STATIC_ASSERT((is_tile<TileI>::value && is_tile<TileJ>::value), "wrong type for the tiles");
#endif
            typedef storage<
#ifdef CXX11_ENABLED
                typename Storage::template type_tt
#else
                base_storage
#endif
                <typename Storage::pointer_type, typename get_tmp_storage_info
                 <typename Storage::meta_data_t::index_type
                  , typename Storage::meta_data_t::layout,
                  typename Storage::meta_data_t::halo_t,
                  typename Storage::meta_data_t::alignment_boundary_t,
#ifdef CXX11_ENABLED
                  Tiles ...
#else
                  TileI, TileJ
#endif
                  >::type, Storage::field_dimensions > > type;
        };
    };

}//namespace gridtools
