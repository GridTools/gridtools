#pragma once
#include "../backend_traits_fwd.hpp"
#include "../mss_functor.hpp"
#include "stencil-composition/backend_host/execute_kernel_functor_host.hpp"
#include "../../storage/meta_storage.hpp"
#include "../tile.hpp"

namespace gridtools{

    template<enumtype::strategy>
    struct strategy_from_id_host;

    /**
       @brief specialization for the \ref gridtools::_impl::Naive strategy
    */
    template<>
    struct strategy_from_id_host< enumtype::Naive>
    {
        // default block size for Naive strategy
        typedef block_size<0,0> block_size_t;
        static const uint_t BI=block_size_t::i_size_t::value;
        static const uint_t BJ=block_size_t::j_size_t::value;
        static const uint_t BK=0;

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
                typedef backend_traits_from_id< BackendId > backend_traits;
                boost::mpl::for_each<iter_range> (mss_functor<MssComponentsArray, Grid, LocalDomainListArray, BackendId, enumtype::Naive> (local_domain_lists, grid,0,0));
            }
        };

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         * @tparam BackendId id of the backend
         */
        template<typename RunFunctorArgs, enumtype::platform BackendId>
        struct mss_loop
        {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), "Internal Error: wrong type");
            template<typename LocalDomain, typename Grid>
            static void run(const LocalDomain& local_domain, const Grid& grid, const uint_t bi, const uint_t bj)
            {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), "Internal Error: wrong type");
                GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), "Internal Error: wrong type");
                typedef backend_traits_from_id< BackendId > backend_traits_t;

                typedef typename backend_traits_t::template execute_traits< RunFunctorArgs >::run_functor_t run_functor_t;

                typedef typename RunFunctorArgs::functor_list_t functor_list_t;
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<functor_list_t>::value==1), "Internal Error: wrong size");

                GRIDPREFIX::execute_kernel_functor_host<RunFunctorArgs>(local_domain, grid)();
            }
        };

        //NOTE: this part is (and should remain) an exact copy-paste in the naive, block, host and cuda versions
        template <typename Index, typename Layout, typename Halo, typename Alignment,
#ifdef CXX11_ENABLED
                  typename ... Tiles
#else
                  typename TileI, typename TileJ
#endif
                  >
        struct get_tmp_storage_info
        {
            GRIDTOOLS_STATIC_ASSERT(is_aligned<Alignment>::type::value,"wrong type");

            GRIDTOOLS_STATIC_ASSERT(is_layout_map<Layout>::value, "wrong type for layout map");
#ifdef CXX11_ENABLED
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_tile<Tiles>::type::value ... ), "wrong type for the tiles");
#else
            GRIDTOOLS_STATIC_ASSERT((is_tile<TileI>::value && is_tile<TileJ>::value), "wrong type for the tiles");
#endif
            GRIDTOOLS_STATIC_ASSERT(is_halo<Halo>::type::value, "wrong type");

            typedef meta_storage_tmp
            <meta_storage_aligned
             <meta_storage_base
              <Index::value, Layout, true>
              , Alignment, Halo
              >
#ifdef CXX11_ENABLED
             , Tiles ...
#else
             , TileI, TileJ
#endif
             > type;
        };

        /**
         * @brief metafunction that returns the storage type for the storage type of the temporaries for this strategy.
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
                 <typename Storage::meta_data_t::index_type, typename Storage::meta_data_t::layout,
                  typename Storage::meta_data_t::halo_t,
                  typename Storage::meta_data_t::alignment_t,
#ifdef CXX11_ENABLED
                  Tiles ...
#else
                  TileI, TileJ
#endif
                  >::type, Storage::field_dimensions > > type;
        };
    };

    /**
       @brief specialization for the \ref gridtools::_impl::Block strategy
       The loops over i and j are split according to the values of BI and BJ
    */
    template<>
    struct strategy_from_id_host <enumtype::Block>
    {
        // default block size for Block strategy
        typedef block_size<GT_DEFAULT_TILE_I,GT_DEFAULT_TILE_J> block_size_t;

        static const uint_t BI=block_size_t::i_size_t::value;
        static const uint_t BJ=block_size_t::j_size_t::value;
        static const uint_t BK=0;


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
                typedef backend_traits_from_id<BackendId> backend_traits;

                uint_t n = grid.i_high_bound() - grid.i_low_bound();
                uint_t m = grid.j_high_bound() - grid.j_low_bound();

                uint_t NBI = n/BI;
                uint_t NBJ = m/BJ;

                #pragma omp parallel
                {
                #pragma omp for nowait
                    for (uint_t bi = 0; bi <= NBI; ++bi) {
                        for (uint_t bj = 0; bj <= NBJ; ++bj) {
                            boost::mpl::for_each<iter_range> (mss_functor<MssComponentsArray, Grid, LocalDomainListArray, BackendId, enumtype::Block> (local_domain_lists, grid,bi,bj));
                        }
                    }
                }
            }
        };

        /**
         * @brief main execution of a mss for a given IJ block. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         * @tparam BackendId id of the backend
         */
        template<typename RunFunctorArgs, enumtype::platform BackendId>
        struct mss_loop
        {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), "Internal Error: wrong type");
            template<typename LocalDomain, typename Grid>
            static void run(const LocalDomain& local_domain, const Grid& grid, const uint_t bi, const uint_t bj)
            {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), "Internal Error: wrong type");
                GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), "Internal Error: wrong type");

                typedef backend_traits_from_id< BackendId > backend_traits_t;

                typedef typename backend_traits_t::template execute_traits< RunFunctorArgs >::run_functor_t run_functor_t;
                typedef typename RunFunctorArgs::functor_list_t functor_list_t;
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<functor_list_t>::value==1), "Internal Error: wrong size");

                uint_t n = grid.i_high_bound() - grid.i_low_bound() ;
                uint_t m = grid.j_high_bound() - grid.j_low_bound() ;

                uint_t NBI = n/BI;
                uint_t NBJ = m/BJ;

                uint_t first_i = bi*BI+grid.i_low_bound();
                uint_t first_j = bj*BJ+grid.j_low_bound();

                uint_t last_i = BI-1;
                uint_t last_j = BJ-1;

                if(bi == NBI && bj == NBJ)
                {
                    last_i = n-NBI*BI;
                    last_j = m-NBJ*BJ;
                }
                else if(bi == NBI)
                {
                    last_i = n-NBI*BI;
                }
                else if(bj == NBJ)
                {
                    last_j = m-NBJ*BJ;
                }

                GRIDPREFIX::execute_kernel_functor_host<RunFunctorArgs>(local_domain, grid, first_i, first_j, last_i, last_j, bi, bj)();
            }
        };


        //NOTE: this part is (and should remain) an exact copy-paste in the naive, block, host and cuda versions
        template <typename Index, typename Layout
                  , typename Halo
#ifdef CXX11_ENABLED
                  , typename ... Tiles
#else
                  , typename TileI, typename TileJ
#endif
                  >
        struct get_tmp_meta_storage
        {
            GRIDTOOLS_STATIC_ASSERT(is_layout_map<Layout>::value, "wrong type for layout map");
#ifdef CXX11_ENABLED
            GRIDTOOLS_STATIC_ASSERT(is_variadic_pack_of(is_tile<Tiles>::type::value ... ), "wrong type for the tiles");
#else
            GRIDTOOLS_STATIC_ASSERT((is_tile<TileI>::value && is_tile<TileJ>::value), "wrong type for the tiles");
#endif
            GRIDTOOLS_STATIC_ASSERT(is_halo<Halo>::type::value, "wrong type");

            typedef meta_storage_tmp
            <meta_storage_aligned
              <meta_storage_base
               <Index::value, Layout, true>
               , aligned<0>
               , Halo
               >
#ifdef CXX11_ENABLED
              , Tiles ...
#else
              , TileI, TileJ
#endif
              > type;
        };

        /**
         * @brief metafunction that returns the storage type for the storage type of the temporaries for this strategy.
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
                <typename Storage::pointer_type, typename get_tmp_meta_storage
                 <typename Storage::meta_data_t::index_type, typename Storage::meta_data_t::layout,
                  typename Storage::meta_data_t::halo_t,
#ifdef CXX11_ENABLED
                  Tiles ...
#else
                  TileI, TileJ
#endif
                  >::type, Storage::field_dimensions > > type;
        };
};

} //namespace gridtools
