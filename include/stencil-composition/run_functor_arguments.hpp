/*
 * run_functor_arguments.h
 *
 *  Created on: Mar 5, 2015
 *      Author: carlosos
 */

#pragma once
#include <boost/static_assert.hpp>
#include "../common/defs.hpp"
#include "block_size.hpp"
#include "local_domain.hpp"
#include "axis.hpp"

namespace gridtools {

    /**
     * @brief type that contains main metadata required to execute a mss kernel. This type will be passed to
     * all functors involved in the execution of the mss
     */
    template<
        enumtype::backend BackendId,                // id of the backend
        typename ProcessingElementsBlockSize,       // block size of grid points updated by computation
                                                    //    in the physical domain
        typename PhysicalDomainBlockSize,           // block size of processing elements (i.e. threads)
                                                    //    taking part in the computation of a physical block size
        typename FunctorList,                       // sequence of functors (one per ESF)
        typename EsfArgsMapSequence,                // map of arg indices from local functor position to a merged
                                                    //    local domain
        typename LoopIntervals,                     // loop intervals
        typename FunctorsMap,                       // functors map
        typename RangeSizes,                        // ranges of each ESF
        typename LocalDomain,                       // local domain type
        typename Coords,                            // the coordinates
        typename ExecutionEngine,                   // the execution engine
        enumtype::strategy StrategyId>              // the strategy id
    struct run_functor_arguments
    {
        GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((is_coordinates<Coords>::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((is_execution_engine<ExecutionEngine>::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((is_block_size<ProcessingElementsBlockSize>::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((is_block_size<PhysicalDomainBlockSize>::value), "Internal Error: invalid type");

        typedef enumtype::enum_type<enumtype::backend, BackendId> backend_id_t;
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

    /**
     * @brief type that contains main metadata required to execute an ESF functor. This type will be passed to
     * all functors involved in the execution of the ESF
     */
    template<typename RunFunctorArguments, typename Index>
    struct esf_arguments
    {
        GRIDTOOLS_STATIC_ASSERT((is_run_functor_arguments<RunFunctorArguments>::value), "Internal Error: invalid type");

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
