#pragma once

#include <gridtools.h>
#include <boost/fusion/include/value_at.hpp>
#include <boost/mpl/has_key.hpp>
#include "../level.h"

#include "backend_traits_cuda.h"
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
        template<typename MssComponentsArray, enumtype::backend BackendId>
        struct fused_mss_loop
        {
            BOOST_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value));
            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<typename MssComponentsArray::elements>::type::value> iter_range;

            template<typename LocalDomainListArray, typename Coords>
            static void run(LocalDomainListArray& local_domain_lists, const Coords& coords)
            {
                typedef backend_traits_from_id< BackendId > backend_traits;
                backend_traits::template for_each<iter_range> (
                    mss_functor<MssComponentsArray, Coords, LocalDomainListArray, BackendId, enumtype::Naive> (local_domain_lists, coords,0,0)
                );
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

    /**
       @brief specialization for the \ref gridtools::_impl::Block strategy
       The loops over i and j are split according to the values of BI and BJ
    */
    template<>
    struct strategy_from_id <enumtype::Block>
    {};

    template <enumtype::backend, uint_t Id>
    struct once_per_block;

}//namespace gridtools
