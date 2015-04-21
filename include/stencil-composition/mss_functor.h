/*
 * mss_functor.h
 *
 *  Created on: Mar 5, 2015
 *      Author: carlosos
 */

#pragma once

#include <common/meta_array.h>
#include "mss_metafunctions.h"
#include "mss_local_domain.h"
#include "mss.h"
#include "axis.h"

namespace gridtools {

    //forward declaration
    template<enumtype::strategy Strategy>
    struct strategy_from_id;

    /**
     * @brief functor that executes all the functors contained within the mss
     * @tparam TMssArray meta array containing all the mss descriptors
     * @tparam Coords coordinates
     * @tparam MssLocalDomainArray sequence of mss local domain (each contains the local domain list of a mss)
     * @tparam BackendId id of backend
     * @tparam StrategyId id of strategy
     */
    template<typename TMssArray, typename Coords, typename MssLocalDomainArray, enumtype::backend BackendId, enumtype::strategy StrategyId>
    struct mss_functor
    {
        BOOST_STATIC_ASSERT((is_sequence_of<MssLocalDomainArray, is_mss_local_domain>::value));
        BOOST_STATIC_ASSERT((is_meta_array_of<TMssArray, is_mss_descriptor>::value));
        BOOST_STATIC_ASSERT((is_coordinates<Coords>::value));

        mss_functor(MssLocalDomainArray& local_domain_lists, const Coords& coords, const int block_idx, const int block_idy) :
            m_local_domain_lists(local_domain_lists), m_coords(coords), m_block_idx(block_idx), m_block_idy(block_idy) {}

        /**
         * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the operations defined on that functor.
         */
        template <typename Index>
        void operator()(Index const& ) const
        {
            typedef typename boost::fusion::result_of::value_at<MssLocalDomainArray, Index>::type mss_local_domain_t;
            BOOST_STATIC_ASSERT((is_mss_local_domain<mss_local_domain_t>::value));
            BOOST_STATIC_ASSERT((Index::value < boost::mpl::size<typename TMssArray::elements>::value));
            typedef typename boost::mpl::at<typename TMssArray::elements, Index>::type MssType;
            typedef typename mss_local_domain_t::LocalDomainList local_domain_list_t;

            typedef typename backend_traits_from_id<BackendId>::
                    template fuse_mss_local_domain_strategy<mss_local_domain_t>::type fused_mss_local_domain_t;

            typedef typename backend_traits_from_id<BackendId>::
                    template args_lookup_map<mss_local_domain_t, fused_mss_local_domain_t>::type args_lookup_map_t;

            local_domain_list_t& local_domain_list = (local_domain_list_t&)boost::fusion::at<Index>(m_local_domain_lists).local_domain_list;

            typedef typename MssType::execution_engine_t ExecutionEngine;

            typedef typename mss_loop_intervals<MssType, Coords>::type LoopIntervals; // List of intervals on which functors are defined
            //wrapping all the template arguments in a single container
            typedef typename boost::mpl::if_<
                    typename boost::mpl::bool_< ExecutionEngine::type::iteration==enumtype::forward >::type,
                    LoopIntervals,
                    typename boost::mpl::reverse<LoopIntervals>::type
            >::type oriented_loop_intervals_t;
            // List of functors to execute (in order)
            typedef typename MssType::functors_list functors_list_t;
            // computed range sizes to know where to compute functot at<i>
            typedef typename MssType::range_sizes range_sizes;
            // Map between interval and actual arguments to pass to Do methods
            typedef typename mss_functor_do_method_lookup_maps<MssType, Coords>::type FunctorsMap;

            // compute the struct with all the type arguments for the run functor
            typedef run_functor_arguments<functors_list_t, oriented_loop_intervals_t, FunctorsMap, range_sizes, local_domain_list_t, Coords, ExecutionEngine, StrategyId> run_functor_args_t;

            typedef backend_traits_from_id< BackendId > backend_traits_t;

            typedef typename backend_traits_t::template execute_traits< run_functor_args_t >::run_functor_t run_functor_t;

            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<functors_list_t>::type::value> iter_range;

            //now the corresponding backend has to execute all the functors of the mss
            backend_traits_from_id<BackendId>::template mss_loop<run_functor_args_t, StrategyId>::template run(local_domain_list, m_coords, m_block_idx, m_block_idy);
        }

    private:
        MssLocalDomainArray& m_local_domain_lists;
        const Coords& m_coords;
        const uint_t m_block_idx, m_block_idy;
    };
} //namespace gridtools
