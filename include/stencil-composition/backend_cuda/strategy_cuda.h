#pragma once

#include <gridtools.h>
#include <boost/fusion/include/value_at.hpp>
#include <boost/mpl/has_key.hpp>
#include "../level.h"

#include "backend_traits_cuda.h"
#include "../../storage/host_tmp_storage.h"
#include "../mss_functor.h"
#include "../sfinae.h"

namespace gridtools{

    template<enumtype::strategy>
    struct strategy_from_id_cuda;

    /**
       @brief specialization for the \ref gridtools::_impl::Naive strategy
    */
    template<>
    struct strategy_from_id_cuda< enumtype::Naive>
    {
        static const uint_t BI=0;
        static const uint_t BJ=0;
        static const uint_t BK=0;

        /**
         * @brief loops over all blocks and execute sequentially all mss functors for each block
         * @tparam MssComponentsArray a meta array with the mss components of all MSS
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

//        //forward declaration
//        template<typename StorageBase,uint_t D,uint_t E,uint_t F,uint_t G,uint_t H,uint_t I >
//        struct host_tmp_storage;

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
//            typedef storage< StorageType > type;
            typedef host_tmp_storage < StorageType, BI, BJ, IMinus, JMinus, IPlus+1, JPlus+1> type;

        };

    };

    /**
       @brief specialization for the \ref gridtools::_impl::Block strategy
       Empty as not used in the CUDA backend
    */
    template<>
    struct strategy_from_id_cuda <enumtype::Block>
    {};

    template <enumtype::backend, uint_t Id>
    struct once_per_block;

}//namespace gridtools
