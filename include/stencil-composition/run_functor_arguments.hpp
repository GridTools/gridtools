/*
  GridTools Libraries

  Copyright (c) 2016, GridTools Consortium
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/
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
#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "caches/cache_metafunctions.hpp"
#include "backend_traits_fwd.hpp"
#include "esf.hpp"
#include "stencil-composition/grid.hpp"
#include "grid_traits.hpp"
#include "backend_ids.hpp"
#include "reductions/reduction_data.hpp"
#include "color.hpp"

namespace gridtools {

    template < typename BackendIds,
        typename LocalDomain,
        typename EsfSequence,
        typename ExtendSizes,
        typename MaxExtent,
        typename CacheSequence,
        typename ProcessingElementsBlockSize,
        typename PhysicalDomainBlockSize,
        typename Grid,
        typename IsReduction,
        typename FunctorReturnType >
    struct iterate_domain_arguments {
        GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), "Internal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), "Iternal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< CacheSequence, is_cache >::value), "Iternal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< EsfSequence, is_esf_descriptor >::value), "Iternal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of< ExtendSizes, is_extent >::value), "Iternal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_block_size< ProcessingElementsBlockSize >::value), "Iternal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_block_size< PhysicalDomainBlockSize >::value), "Iternal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "Iternal Error: wrong type");
        GRIDTOOLS_STATIC_ASSERT(
            (IsReduction::value == true || IsReduction::value == false), "Internal Error: wrong type");

        typedef BackendIds backend_ids_t;
        typedef LocalDomain local_domain_t;
        typedef CacheSequence cache_sequence_t;
        typedef EsfSequence esf_sequence_t;
        typedef ExtendSizes extent_sizes_t;
        typedef MaxExtent max_extent_t;
        typedef ProcessingElementsBlockSize processing_elements_block_size_t;
        typedef PhysicalDomainBlockSize physical_domain_block_size_t;
        typedef Grid grid_t;
        static const bool s_is_reduction = IsReduction::value;
        typedef IsReduction is_reduction_t;
        typedef FunctorReturnType functor_return_type_t;
    };

    template < typename T >
    struct is_iterate_domain_arguments : boost::mpl::false_ {};

    template < typename BackendIds,
        typename LocalDomain,
        typename EsfSequence,
        typename ExtendSizes,
        typename MaxExtent,
        typename CacheSequence,
        typename ProcessingElementsBlockSize,
        typename PhysicalDomainBlockSize,
        typename Grid,
        typename IsReduction,
        typename FunctorReturnType >
    struct is_iterate_domain_arguments< iterate_domain_arguments< BackendIds,
        LocalDomain,
        EsfSequence,
        ExtendSizes,
        MaxExtent,
        CacheSequence,
        ProcessingElementsBlockSize,
        PhysicalDomainBlockSize,
        Grid,
        IsReduction,
        FunctorReturnType > > : boost::mpl::true_ {};

    /**
     * @brief type that contains main metadata required to execute a mss kernel. This type will be passed to
     * all functors involved in the execution of the mss
     */
    template < typename BackendIds,           // id of the different backends
        typename ProcessingElementsBlockSize, // block size of grid points updated by computation
                                              //    in the physical domain
        typename PhysicalDomainBlockSize,     // block size of processing elements (i.e. threads)
                                              //    taking part in the computation of a physical block size
        typename FunctorList,                 // sequence of functors (one per ESF)
        typename EsfSequence,                 // sequence of ESF
        typename EsfArgsMapSequence,          // map of arg indices from local functor position to a merged
                                              //    local domain
        typename LoopIntervals,               // loop intervals
        typename FunctorsMap,                 // functors map
        typename ExtendSizes,                 // extents of each ESF
        typename LocalDomain,                 // local domain type
        typename CacheSequence,               // sequence of user specified caches
        typename IsIndependentSeq, // sequence of boolenans (one per functor), stating if it is contained in a
                                   // "make_independent" construct
        typename Grid,             // the grid
        typename ExecutionEngine,  // the execution engine
        typename IsReduction,      // boolean stating if the operation to be applied at mss is a reduction
        typename ReductionData,    // return type of functors of a mss: return type of reduction operations,
                                   //        otherwise void
        typename Color             // current color execution (not used for rectangular grids, or grids that dont have
                                   // concept of a color
        >
    struct run_functor_arguments {
        GRIDTOOLS_STATIC_ASSERT((is_backend_ids< BackendIds >::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((is_local_domain< LocalDomain >::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((is_grid< Grid >::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((is_execution_engine< ExecutionEngine >::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((is_block_size< ProcessingElementsBlockSize >::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((is_block_size< PhysicalDomainBlockSize >::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT(
            (is_sequence_of< EsfSequence, is_esf_descriptor >::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((is_reduction_data< ReductionData >::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT((is_color_type< Color >::value), "Internal Error: invalid type");
        GRIDTOOLS_STATIC_ASSERT(
            (IsReduction::value == true || IsReduction::value == false), "Internal Error: wrong type");

        typedef BackendIds backend_ids_t;
        typedef ProcessingElementsBlockSize processing_elements_block_size_t;
        typedef PhysicalDomainBlockSize physical_domain_block_size_t;
        typedef FunctorList functor_list_t;
        typedef EsfSequence esf_sequence_t;
        typedef EsfArgsMapSequence esf_args_map_sequence_t;
        typedef LoopIntervals loop_intervals_t;
        typedef FunctorsMap functors_map_t;
        typedef ExtendSizes extent_sizes_t;
        typedef typename boost::mpl::fold< extent_sizes_t,
            typename grid_traits_from_id< backend_ids_t::s_grid_type_id >::null_extent_t,
            enclosing_extent< boost::mpl::_1, boost::mpl::_2 > >::type max_extent_t;
        typedef LocalDomain local_domain_t;
        typedef CacheSequence cache_sequence_t;
        typedef IsIndependentSeq async_esf_map_t;
        typedef typename backend_traits_from_id< backend_ids_t::s_backend_id >::template select_iterate_domain<
            iterate_domain_arguments< BackendIds,
                LocalDomain,
                EsfSequence,
                ExtendSizes,
                max_extent_t,
                CacheSequence,
                ProcessingElementsBlockSize,
                PhysicalDomainBlockSize,
                Grid,
                IsReduction,
                typename ReductionData::reduction_type_t > >::type iterate_domain_t;
        typedef Grid grid_t;
        typedef ExecutionEngine execution_type_t;
        static const enumtype::strategy s_strategy_id = backend_ids_t::s_strategy_id;
        static const bool s_is_reduction = IsReduction::value;
        typedef IsReduction is_reduction_t;
        typedef ReductionData reduction_data_t;
        typedef Color color_t;
    };

    template < typename T >
    struct is_run_functor_arguments : boost::mpl::false_ {};

    template < typename BackendIds,
        typename ProcessingElementsBlockSize,
        typename PhysicalDomainBlockSize,
        typename FunctorList,
        typename EsfSequence,
        typename EsfArgsMapSequence,
        typename LoopIntervals,
        typename FunctorsMap,
        typename ExtendSizes,
        typename LocalDomain,
        typename CacheSequence,
        typename IsIndependentSequence,
        typename Grid,
        typename ExecutionEngine,
        typename IsReduction,
        typename ReductionData,
        typename Color >
    struct is_run_functor_arguments< run_functor_arguments< BackendIds,
        ProcessingElementsBlockSize,
        PhysicalDomainBlockSize,
        FunctorList,
        EsfSequence,
        EsfArgsMapSequence,
        LoopIntervals,
        FunctorsMap,
        ExtendSizes,
        LocalDomain,
        CacheSequence,
        IsIndependentSequence,
        Grid,
        ExecutionEngine,
        IsReduction,
        ReductionData,
        Color > > : boost::mpl::true_ {};

    /**
     * @brief type that contains main metadata required to execute an ESF functor. This type will be passed to
     * all functors involved in the execution of the ESF
     */
    template < typename RunFunctorArguments, typename Index >
    struct esf_arguments {
        GRIDTOOLS_STATIC_ASSERT(
            (is_run_functor_arguments< RunFunctorArguments >::value), "Internal Error: invalid type");

        typedef typename boost::mpl::at< typename RunFunctorArguments::functor_list_t, Index >::type functor_t;
        typedef typename boost::mpl::at< typename RunFunctorArguments::esf_args_map_sequence_t, Index >::type
            esf_args_map_t;
        typedef typename boost::mpl::at< typename RunFunctorArguments::extent_sizes_t, Index >::type extent_t;
        typedef typename boost::mpl::at< typename RunFunctorArguments::functors_map_t, Index >::type interval_map_t;
        typedef typename boost::mpl::at< typename RunFunctorArguments::esf_sequence_t, Index >::type esf_t;

        // global (to the mss) sequence_of_is_independent_t map (not local to the esf)
        typedef typename RunFunctorArguments::async_esf_map_t async_esf_map_t;

        typedef typename RunFunctorArguments::is_reduction_t is_reduction_t;
        typedef typename index_to_level<
            typename boost::mpl::deref< typename boost::mpl::find_if< typename RunFunctorArguments::loop_intervals_t,
                boost::mpl::has_key< interval_map_t, boost::mpl::_1 > >::type >::type::first >::type first_hit_t;
        typedef typename RunFunctorArguments::reduction_data_t reduction_data_t;
    };

    template < typename T >
    struct is_esf_arguments : boost::mpl::false_ {};

    template < typename RunFunctorArguments, typename Index >
    struct is_esf_arguments< esf_arguments< RunFunctorArguments, Index > > : boost::mpl::true_ {};

} // namespace gridtools
