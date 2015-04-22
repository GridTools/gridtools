/*
 * run_functor_arguments.h
 *
 *  Created on: Mar 5, 2015
 *      Author: carlosos
 */

#pragma once

namespace gridtools {

    /** The following struct is defined here since the current version of NVCC does not accept local types to be used as template arguments of __global__ functions \todo move inside backend::run()*/
    template<
        typename FunctorList,
        typename EsfArgsMap,
        typename LoopIntervals,
        typename FunctorsMap,
        typename RangeSizes,
        typename LocalDomainList,
        typename Coords,
        typename ExecutionEngine,
        enumtype::strategy StrategyId>
    struct run_functor_arguments
    {
        typedef FunctorList functor_list_t;
        typedef EsfArgsMap esf_args_map_t;
        typedef LoopIntervals loop_intervals_t;
        typedef FunctorsMap functors_map_t;
        typedef RangeSizes range_sizes_t;
        typedef LocalDomainList local_domain_list_t;
        typedef Coords coords_t;
        typedef ExecutionEngine execution_type_t;
        static const enumtype::strategy s_strategy_id=StrategyId;
    };

    template<typename T> struct is_run_functor_arguments : boost::mpl::false_{};

    template<
        typename FunctorList,
        typename EsfArgsMap,
        typename LoopIntervals,
        typename FunctorsMap,
        typename RangeSizes,
        typename LocalDomainList,
        typename Coords,
        typename ExecutionEngine,
        enumtype::strategy StrategyId>
    struct is_run_functor_arguments<
        run_functor_arguments<
            FunctorList,
            EsfArgsMap,
            LoopIntervals,
            FunctorsMap,
            RangeSizes,
            LocalDomainList,
            Coords,
            ExecutionEngine,
            StrategyId
        >
    > : boost::mpl::true_{};

} // namespace gridtools
