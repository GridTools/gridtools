#pragma once
#include "../backend_traits_fwd.hpp"
#include "../mss_functor.hpp"
#include "execute_kernel_functor_host.hpp"

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
        template<typename MssComponentsArray, enumtype::backend BackendId>
        struct fused_mss_loop
        {
            GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value), "Internal Error: wrong type");
            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename MssComponentsArray::elements>::type::value> iter_range;

            template<typename LocalDomainListArray, typename Coords>
            static void run(LocalDomainListArray& local_domain_lists, const Coords& coords)
            {
                typedef backend_traits_from_id< BackendId > backend_traits;
                gridtools::for_each<iter_range> (mss_functor<MssComponentsArray, Coords, LocalDomainListArray, BackendId, enumtype::Naive> (local_domain_lists, coords,0,0));
            }
        };

        /**
         * @brief main execution of a mss. Defines the IJ loop bounds of this particular block
         * and sequentially executes all the functors in the mss
         * @tparam RunFunctorArgs run functor arguments
         * @tparam BackendId id of the backend
         */
        template<typename RunFunctorArgs, enumtype::backend BackendId>
        struct mss_loop
        {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), "Internal Error: wrong type");
            template<typename LocalDomain, typename Coords>
            static void run(const LocalDomain& local_domain, const Coords& coords, const uint_t bi, const uint_t bj)
            {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), "Internal Error: wrong type");
                GRIDTOOLS_STATIC_ASSERT((is_coordinates<Coords>::value), "Internal Error: wrong type");
                typedef backend_traits_from_id< BackendId > backend_traits_t;

                typedef typename backend_traits_t::template execute_traits< RunFunctorArgs >::run_functor_t run_functor_t;

                typedef typename RunFunctorArgs::functor_list_t functor_list_t;
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<functor_list_t>::value==1), "Internal Error: wrong size");

                execute_kernel_functor_host<RunFunctorArgs>(local_domain, coords)();
            }
        };

        /**
         * @brief metafunction that returns the storage type for the storage type of the temporaries for this strategy.
         * with the naive algorithms, the temporary storages are like the non temporary ones
         */
         template <typename StorageType,
                   uint_t BI,
                   uint_t BJ,
                   uint_t IMinus,
                   uint_t JMinus,
                   uint_t IPlus,
                   uint_t JPlus>
         struct get_tmp_storage
         {
 //#warning "the temporary fields you specified will be allocated (like the non-temporary ones). To avoid this use the Block strategy instead of the Naive."
             typedef storage< StorageType > type;
         };

    };

    //forward declaration
    template<typename StorageBase,uint_t D,uint_t E,uint_t F,uint_t G,uint_t H,uint_t I >
    struct host_tmp_storage;

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
        template<typename MssComponentsArray, enumtype::backend BackendId>
        struct fused_mss_loop
        {
            GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value), "Internal Error: wrong type");
            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename MssComponentsArray::elements>::type::value> iter_range;

            template<typename LocalDomainListArray, typename Coords>
            static void run(LocalDomainListArray& local_domain_lists, const Coords& coords)
            {
                GRIDTOOLS_STATIC_ASSERT((is_coordinates<Coords>::value), "Internal Error: wrong type");
                typedef backend_traits_from_id<BackendId> backend_traits;

                uint_t n = coords.i_high_bound() - coords.i_low_bound();
                uint_t m = coords.j_high_bound() - coords.j_low_bound();

                uint_t NBI = n/BI;
                uint_t NBJ = m/BJ;

                #pragma omp parallel
                {
                #pragma omp for nowait
                    for (uint_t bi = 0; bi <= NBI; ++bi) {
                        for (uint_t bj = 0; bj <= NBJ; ++bj) {
                            gridtools::for_each<iter_range> (mss_functor<MssComponentsArray, Coords, LocalDomainListArray, BackendId, enumtype::Block> (local_domain_lists, coords,bi,bj));
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
        template<typename RunFunctorArgs, enumtype::backend BackendId>
        struct mss_loop
        {
            GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value), "Internal Error: wrong type");
            template<typename LocalDomain, typename Coords>
            static void run(const LocalDomain& local_domain, const Coords& coords, const uint_t bi, const uint_t bj)
            {
                GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), "Internal Error: wrong type");
                GRIDTOOLS_STATIC_ASSERT((is_coordinates<Coords>::value), "Internal Error: wrong type");

                typedef backend_traits_from_id< BackendId > backend_traits_t;

                typedef typename backend_traits_t::template execute_traits< RunFunctorArgs >::run_functor_t run_functor_t;
                typedef typename RunFunctorArgs::functor_list_t functor_list_t;
                GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<functor_list_t>::value==1), "Internal Error: wrong size");

                uint_t n = coords.i_high_bound() - coords.i_low_bound() ;
                uint_t m = coords.j_high_bound() - coords.j_low_bound() ;

                uint_t NBI = n/BI;
                uint_t NBJ = m/BJ;

                uint_t first_i = bi*BI+coords.i_low_bound();
                uint_t first_j = bj*BJ+coords.j_low_bound();

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

                execute_kernel_functor_host<RunFunctorArgs>(local_domain, coords, first_i, first_j, last_i, last_j, bi, bj)();
            }
        };

        /**
         * @brief metafunction that returns the storage type for the storage type of the temporaries for this strategy.
         */
        template <typename StorageBase ,
                  uint_t BI,
                  uint_t BJ,
                  uint_t IMinus,
                  uint_t JMinus,
                  uint_t IPlus,
                  uint_t JPlus>
        struct get_tmp_storage
        {
            typedef host_tmp_storage < StorageBase, BI, BJ, IMinus, JMinus, IPlus+1, JPlus+1> type;
        };
    };

} //namespace gridtools
