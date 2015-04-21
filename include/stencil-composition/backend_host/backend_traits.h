#pragma once

#include <gridtools.h>
#include <boost/fusion/include/value_at.hpp>
#include <boost/mpl/has_key.hpp>
#include "../level.h"

#include "backend_traits_host.h"
#include "../mss_functor.h"
#include "../sfinae.h"

/**
   @file

   \brief This class contains the traits which are used in backand.h
*/

namespace gridtools{
    /**
       @brief specialization for the \ref gridtools::_impl::Naive strategy
       A single loop spans all three directions, i, j and k
    */
    template<>
    struct strategy_from_id< enumtype::Naive>
    {
        static const uint_t BI=0;
        static const uint_t BJ=0;
        static const uint_t BK=0;

        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam TMssArray a meta array with all the mss descriptors
         * @tparam BackendId id of the backend
         */
        template<typename TMssArray, enumtype::backend BackendId>
        struct fused_mss_loop
        {
            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename TMssArray::elements>::type::value> iter_range;

            template<typename LocalDomainListArray, typename Coords>
            static void run(LocalDomainListArray& local_domain_lists, const Coords& coords)
            {
                typedef backend_traits_from_id< BackendId > backend_traits;
                backend_traits::template for_each<iter_range> (mss_functor<TMssArray, Coords, LocalDomainListArray, BackendId, enumtype::Naive> (local_domain_lists, coords,0,0));
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
            BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value));
            template<typename LocalDomainList, typename Coords>
            static void run(LocalDomainList& local_domain_list, const Coords& coords, const uint_t bi, const uint_t bj)
            {
                BOOST_STATIC_ASSERT((is_coordinates<Coords>::value));
                typedef backend_traits_from_id< BackendId > backend_traits_t;

                typedef typename backend_traits_t::template execute_traits< RunFunctorArgs >::run_functor_t run_functor_t;

                typedef typename RunFunctorArgs::functor_list_t functor_list_t;

                typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<functor_list_t>::type::value> iter_range;

                backend_traits_t::template for_each< iter_range >(run_functor_t(local_domain_list, coords));
            }
        };

        //with the naive algorithms, the temporary storages are like the non temporary ones
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
    struct strategy_from_id <enumtype::Block>
    {
        static const uint_t BI=GT_DEFAULT_TILE;
        static const uint_t BJ=GT_DEFAULT_TILE;
        static const uint_t BK=0;

        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam TMssArray a meta array with all the mss descriptors
         * @tparam BackendId id of the backend
         */
        template<typename TMssArray, enumtype::backend BackendId>
        struct fused_mss_loop
        {
            BOOST_STATIC_ASSERT((is_meta_array_of<TMssArray, is_mss_descriptor>::value));
            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename TMssArray::elements>::type::value> iter_range;

            template<typename LocalDomainListArray, typename Coords>
            static void run(LocalDomainListArray& local_domain_lists, const Coords& coords)
            {
                BOOST_STATIC_ASSERT((is_coordinates<Coords>::value));
                typedef backend_traits_from_id<BackendId> backend_traits;

                //TODO consider the largest ij range of all mss to compute number of blocks?
                uint_t n = coords.i_high_bound() - coords.i_low_bound();
                uint_t m = coords.j_high_bound() - coords.j_low_bound();

                uint_t NBI = n/BI;
                uint_t NBJ = m/BJ;

                #pragma omp parallel
                {
                    backend_traits::set_thread_id();
                #pragma omp for nowait
                    for (uint_t bi = 0; bi <= NBI; ++bi) {
                        for (uint_t bj = 0; bj <= NBJ; ++bj) {
                            backend_traits::template for_each<iter_range> (mss_functor<TMssArray, Coords, LocalDomainListArray, BackendId, enumtype::Block> (local_domain_lists, coords,bi,bj));
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
            BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArgs>::value));
            template<typename LocalDomainList, typename Coords>
            static void run(LocalDomainList& local_domain_list, const Coords& coords, const uint_t bi, const uint_t bj)
            {
                BOOST_STATIC_ASSERT((is_coordinates<Coords>::value));
                typedef backend_traits_from_id< BackendId > backend_traits_t;

                typedef typename backend_traits_t::template execute_traits< RunFunctorArgs >::run_functor_t run_functor_t;
                typedef typename RunFunctorArgs::functor_list_t functor_list_t;

                typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<functor_list_t>::type::value> iter_range;

                typedef typename boost::mpl::at<typename RunFunctorArgs::range_sizes_t, typename boost::mpl::back<iter_range>::type >::type range_t;

                uint_t n = coords.i_high_bound() + range_t::iplus::value - coords.i_low_bound() + range_t::iminus::value;
                uint_t m = coords.j_high_bound() + range_t::jplus::value - coords.j_low_bound() + range_t::jminus::value;

                uint_t NBI = n/BI;
                uint_t NBJ = m/BJ;

                uint_t _starti = bi*BI+coords.i_low_bound();
                uint_t _startj = bj*BJ+coords.j_low_bound();

                uint_t block_size_i = BI-1;
                uint_t block_size_j = BJ-1;


                if(bi == NBI && bj == NBJ)
                {
                    block_size_i = n-NBI*BI;
                    block_size_j = m-NBJ*BJ;
                }
                else if(bi == NBI)
                {
                    block_size_i = n-NBI*BI;
                }
                else if(bj == NBJ)
                {
                    block_size_j = m-NBJ*BJ;
                }
                backend_traits_t::template for_each< iter_range >(run_functor_t(local_domain_list, coords, _starti, _startj, block_size_i, block_size_j, bi, bj));
            }
        };


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

    template <enumtype::backend, uint_t Id>
    struct once_per_block;

}//namespace gridtools
