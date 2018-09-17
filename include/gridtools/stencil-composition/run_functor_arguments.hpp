/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
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

#pragma once

#include <boost/mpl/at.hpp>
#include <boost/mpl/fold.hpp>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "../common/generic_metafunctions/meta.hpp"
#include "./backend_ids.hpp"
#include "./backend_traits_fwd.hpp"
#include "./caches/cache_traits.hpp"
#include "./color.hpp"
#include "./extent.hpp"
#include "./grid.hpp"
#include "./grid_traits_fwd.hpp"
#include "./local_domain.hpp"
#include "./loop_interval.hpp"
#include "./reductions/reduction_data.hpp"

namespace gridtools {

    template <typename BackendIds,
        typename LocalDomain,
        typename EsfSequence,
        typename ExtentSizes,
        typename MaxExtent,
        typename CacheSequence,
        typename Grid,
        typename IsReduction = std::false_type,
        typename FunctorReturnType = notype>
    struct iterate_domain_arguments {

        GRIDTOOLS_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<CacheSequence, is_cache>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<EsfSequence, is_esf_descriptor>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<ExtentSizes, is_extent>::value),
            "There seems to be a stage in the computation which does not contain any output field. Check that at least "
            "one accessor in each stage is defined as \'inout\'");
        GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((IsReduction::value == true || IsReduction::value == false), GT_INTERNAL_ERROR);

        typedef BackendIds backend_ids_t;
        typedef LocalDomain local_domain_t;
        typedef CacheSequence cache_sequence_t;
        typedef EsfSequence esf_sequence_t;
        typedef ExtentSizes extent_sizes_t;
        typedef MaxExtent max_extent_t;
        typedef Grid grid_t;
        static const bool s_is_reduction = IsReduction::value;
        typedef IsReduction is_reduction_t;
        typedef FunctorReturnType functor_return_type_t;
    };

    template <class T>
    GT_META_DEFINE_ALIAS(is_iterate_domain_arguments, meta::is_instantiation_of, (iterate_domain_arguments, T));

    /**
     * @brief type that contains main metadata required to execute a mss kernel. This type will be passed to
     * all functors involved in the execution of the mss
     */
    template <typename BackendIds, // id of the different backends
        typename EsfSequence,      // sequence of ESF
        typename LoopIntervals,    // loop intervals
        typename ExtentSizes,      // extents of each ESF
        typename LocalDomain,      // local domain type
        typename CacheSequence,    // sequence of user specified caches
        typename Grid,             // the grid
        typename ExecutionEngine,  // the execution engine
        typename IsReduction,      // boolean stating if the operation to be applied at mss is a reduction
        typename ReductionData     // return type of functors of a mss: return type of reduction operations,
                                   //        otherwise void
        >
    struct run_functor_arguments {
        GRIDTOOLS_STATIC_ASSERT((is_backend_ids<BackendIds>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_local_domain<LocalDomain>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_grid<Grid>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_execution_engine<ExecutionEngine>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((is_sequence_of<EsfSequence, is_esf_descriptor>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((meta::all_of<is_loop_interval, LoopIntervals>::value), GT_INTERNAL_ERROR);

        GRIDTOOLS_STATIC_ASSERT((is_reduction_data<ReductionData>::value), GT_INTERNAL_ERROR);
        GRIDTOOLS_STATIC_ASSERT((IsReduction::value == true || IsReduction::value == false), GT_INTERNAL_ERROR);

        typedef BackendIds backend_ids_t;
        typedef EsfSequence esf_sequence_t;
        typedef LoopIntervals loop_intervals_t;
        typedef ExtentSizes extent_sizes_t;
        typedef typename boost::mpl::
            fold<extent_sizes_t, extent<>, enclosing_extent_2<boost::mpl::_1, boost::mpl::_2>>::type max_extent_t;
        typedef LocalDomain local_domain_t;
        typedef CacheSequence cache_sequence_t;
        typedef Grid grid_t;
        typedef ExecutionEngine execution_type_t;
        using strategy_type = typename backend_ids_t::strategy_id_t;
        static constexpr bool s_is_reduction = IsReduction::value;
        typedef IsReduction is_reduction_t;
        typedef ReductionData reduction_data_t;
    };

    template <class T>
    GT_META_DEFINE_ALIAS(is_run_functor_arguments, meta::is_instantiation_of, (run_functor_arguments, T));
} // namespace gridtools
