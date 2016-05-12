/*
 * mss_functor.h
 *
 *  Created on: Mar 5, 2015
 *      Author: carlosos
 */

#pragma once

#include "common/meta_array.hpp"
#include "mss_metafunctions.hpp"
#include "mss_local_domain.hpp"
#include "mss.hpp"
#include "axis.hpp"
#include "mss_components_metafunctions.hpp"

namespace gridtools {

    /**
     * @brief functor that executes all the functors contained within the mss
     * @tparam TMssArray meta array containing all the mss descriptors
     * @tparam Coords coordinates
     * @tparam MssLocalDomainArray sequence of mss local domain (each contains the local domain list of a mss)
     * @tparam BackendId id of backend
     * @tparam StrategyId id of strategy
     */
    template<
        typename MssComponentsArray, typename Coords, typename MssLocalDomainArray,
        enumtype::backend BackendId, enumtype::strategy StrategyId>
    struct mss_functor
    {
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<MssLocalDomainArray, is_mss_local_domain>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_meta_array_of<MssComponentsArray, is_mss_components>::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_coordinates<Coords>::value), "Internal Error: wrong type");

        mss_functor(MssLocalDomainArray& local_domain_lists, const Coords& coords, const int block_idx, const int block_idy) :
            m_local_domain_lists(local_domain_lists), m_coords(coords), m_block_idx(block_idx), m_block_idy(block_idy) {}

        /**
         * \brief given the index of a functor in the functors list ,it calls a kernel on the GPU executing the operations defined on that functor.
         */
        template <typename Index>
        void operator()(Index const& ) const
        {
            typedef typename boost::fusion::result_of::value_at<MssLocalDomainArray, Index>::type mss_local_domain_t;
            GRIDTOOLS_STATIC_ASSERT((is_mss_local_domain<mss_local_domain_t>::value), "Internal Error: wrong type");
            GRIDTOOLS_STATIC_ASSERT((Index::value < boost::mpl::size<typename MssComponentsArray::elements>::value),
                                    "Internal Error");
            typedef typename boost::mpl::at<typename MssComponentsArray::elements, Index>::type mss_components_t;
            typedef typename mss_local_domain_list<mss_local_domain_t>::type local_domain_list_t;
            typedef typename mss_local_domain_esf_args_map<mss_local_domain_t>::type local_domain_esf_args_map_t;

            GRIDTOOLS_STATIC_ASSERT((boost::mpl::size<local_domain_list_t>::value==1), "Internal Error: wrong size");
            typedef typename boost::mpl::back<local_domain_list_t>::type local_domain_t;
            local_domain_list_t& local_domain_list = (local_domain_list_t&)boost::fusion::at<Index>(m_local_domain_lists).local_domain_list;
            local_domain_t& local_domain = (local_domain_t&)boost::fusion::at<boost::mpl::int_<0> >(local_domain_list);

            typedef typename mss_components_t::execution_engine_t ExecutionEngine;

            typedef typename mss_loop_intervals<mss_components_t, Coords>::type LoopIntervals; // List of intervals on which functors are defined
            //wrapping all the template arguments in a single container
            typedef typename boost::mpl::if_<
                    typename boost::mpl::bool_< ExecutionEngine::type::iteration==enumtype::forward >::type,
                    LoopIntervals,
                    typename boost::mpl::reverse<LoopIntervals>::type
            >::type oriented_loop_intervals_t;
            // List of functors to execute (in order)
            typedef typename mss_components_t::functors_list_t functors_list_t;
            // sequence of esf descriptors contained in this mss
            typedef typename mss_components_t::linear_esf_t esf_sequence_t;
            // computed range sizes to know where to compute functot at<i>
            typedef typename mss_components_t::range_sizes_t range_sizes;
            // Map between interval and actual arguments to pass to Do methods
            typedef typename mss_functor_do_method_lookup_maps<mss_components_t, Coords>::type FunctorsMap;

            typedef backend_traits_from_id< BackendId > backend_traits_t;

            typedef typename backend_traits_t::template get_block_size<StrategyId>::type block_size_t;
            // compute the struct with all the type arguments for the run functor
            typedef run_functor_arguments<
                BackendId,
                block_size_t,
                block_size_t,
                functors_list_t,
                esf_sequence_t,
                local_domain_esf_args_map_t,
                oriented_loop_intervals_t,
                FunctorsMap,
                range_sizes,
                local_domain_t,
                typename mss_components_t::cache_sequence_t,
                Coords,
                ExecutionEngine,
                StrategyId
            > run_functor_args_t;

            typedef boost::mpl::range_c<uint_t, 0, boost::mpl::size<functors_list_t>::type::value> iter_range;

            //now the corresponding backend has to execute all the functors of the mss
            backend_traits_from_id<BackendId>::template mss_loop<run_functor_args_t, StrategyId>::template run(local_domain, m_coords, m_block_idx, m_block_idy);
        }

    private:
        MssLocalDomainArray& m_local_domain_lists;
        const Coords& m_coords;
        const uint_t m_block_idx, m_block_idy;
    };
} //namespace gridtools
