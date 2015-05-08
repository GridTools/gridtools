/*
 * run_functor_arguments.h
 *
 *  Created on: Mar 5, 2015
 *      Author: carlosos
 */

#pragma once
#include <boost/static_assert.hpp>
#include "../common/defs.h"
#include "block_size.h"
#include "local_domain.h"
#include "axis.h"

namespace gridtools {

    /** The following struct is defined here since the current version of NVCC does not accept local types to be used as template arguments of __global__ functions \todo move inside backend::run()*/
    template<
        enumtype::backend BackendId,
        typename ProcessingElementsBlockSize,
        typename PhysicalDomainBlockSize,
        typename FunctorList,
        typename EsfArgsMapSequence,
        typename LoopIntervals,
        typename FunctorsMap,
        typename RangeSizes,
        typename LocalDomain,
        typename Coords,
        typename ExecutionEngine,
        enumtype::strategy StrategyId>
    struct run_functor_arguments
    {
        BOOST_STATIC_ASSERT((is_local_domain<LocalDomain>::value));
        BOOST_STATIC_ASSERT((is_coordinates<Coords>::value));
        BOOST_STATIC_ASSERT((is_execution_engine<ExecutionEngine>::value));
        BOOST_STATIC_ASSERT((is_block_size<ProcessingElementsBlockSize>::value));
        BOOST_STATIC_ASSERT((is_block_size<PhysicalDomainBlockSize>::value));

        typedef boost::mpl::integral_c<enumtype::backend, BackendId> backend_id_t;
        typedef ProcessingElementsBlockSize processing_elements_block_size_t;
        typedef PhysicalDomainBlockSize physical_domain_block_size_t;
        typedef FunctorList functor_list_t;
        typedef EsfArgsMapSequence esf_args_map_sequence_t;
        typedef LoopIntervals loop_intervals_t;
        typedef FunctorsMap functors_map_t;
        typedef RangeSizes range_sizes_t;
        typedef LocalDomain local_domain_t;
        typedef typename backend_traits_from_id<backend_id_t::value>::
                template select_iterate_domain<local_domain_t>::type iterate_domain_t;
        typedef Coords coords_t;
        typedef ExecutionEngine execution_type_t;
        static const enumtype::strategy s_strategy_id=StrategyId;
    };

    template<typename T> struct is_run_functor_arguments : boost::mpl::false_{};

    template<
        enumtype::backend BackendId,
        typename ProcessingElementsBlockSize,
        typename PhysicalDomainBlockSize,
        typename FunctorList,
        typename EsfArgsMapSequence,
        typename LoopIntervals,
        typename FunctorsMap,
        typename RangeSizes,
        typename LocalDomain,
        typename Coords,
        typename ExecutionEngine,
        enumtype::strategy StrategyId>
    struct is_run_functor_arguments<
        run_functor_arguments<
            BackendId,
            ProcessingElementsBlockSize,
            PhysicalDomainBlockSize,
            FunctorList,
            EsfArgsMapSequence,
            LoopIntervals,
            FunctorsMap,
            RangeSizes,
            LocalDomain,
            Coords,
            ExecutionEngine,
            StrategyId
        >
    > : boost::mpl::true_{};

    template<typename RunFunctorArguments, typename Index>
    struct esf_arguments
    {
        BOOST_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value));

        typedef typename boost::mpl::at<typename RunFunctorArguments::functor_list_t, Index>::type functor_t;
        typedef typename boost::mpl::at<typename RunFunctorArguments::esf_args_map_sequence_t, Index>::type esf_args_map_t;
        typedef typename boost::mpl::at<typename RunFunctorArguments::range_sizes_t, Index>::type range_t;
        typedef typename boost::mpl::at<typename RunFunctorArguments::functors_map_t, Index>::type interval_map_t;
        typedef typename index_to_level<
            typename boost::mpl::deref<
                typename boost::mpl::find_if<
                    typename RunFunctorArguments::loop_intervals_t,
                    boost::mpl::has_key<interval_map_t, boost::mpl::_1>
                    >::type
                >::type::first
            >::type first_hit_t;
    };

    template<typename T> struct is_esf_arguments : boost::mpl::false_{};

    template<typename RunFunctorArguments, typename Index>
    struct is_esf_arguments<esf_arguments<RunFunctorArguments, Index> > :
        boost::mpl::true_{};

} // namespace gridtools
