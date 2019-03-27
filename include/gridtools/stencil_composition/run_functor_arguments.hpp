/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <boost/mpl/at.hpp>
#include <boost/mpl/fold.hpp>

#include "../common/defs.hpp"
#include "../common/generic_metafunctions/is_sequence_of.hpp"
#include "../meta/is_instantiation_of.hpp"
#include "../meta/logical.hpp"
#include "./caches/cache_traits.hpp"
#include "./color.hpp"
#include "./extent.hpp"
#include "./grid.hpp"
#include "./local_domain.hpp"
#include "./loop_interval.hpp"

namespace gridtools {

    template <typename Target, typename LocalDomain, typename EsfSequence, typename CacheSequence, typename Grid>
    struct iterate_domain_arguments {

        GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_cache, CacheSequence>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_esf_descriptor, EsfSequence>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);

        typedef Target target_t;
        typedef LocalDomain local_domain_t;
        typedef CacheSequence cache_sequence_t;
        typedef EsfSequence esf_sequence_t;
        typedef typename LocalDomain::max_extent_for_tmp_t max_extent_for_tmp_t;
        typedef Grid grid_t;
    };

    template <class T>
    GT_META_DEFINE_ALIAS(is_iterate_domain_arguments, meta::is_instantiation_of, (iterate_domain_arguments, T));

    /**
     * @brief type that contains main metadata required to execute a mss kernel. This type will be passed to
     * all functors involved in the execution of the mss
     */
    template <typename Target,   // id of the different backends
        typename EsfSequence,    // sequence of ESF
        typename LoopIntervals,  // loop intervals
        typename LocalDomain,    // local domain type
        typename CacheSequence,  // sequence of user specified caches
        typename Grid,           // the grid
        typename ExecutionEngine // the execution engine
        >
    struct run_functor_arguments {
        GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_grid<Grid>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_execution_engine<ExecutionEngine>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_esf_descriptor, EsfSequence>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_loop_interval, LoopIntervals>::value), GT_INTERNAL_ERROR);

        typedef Target target_t;
        typedef EsfSequence esf_sequence_t;
        typedef LoopIntervals loop_intervals_t;

      private:
        using all_stage_groups_t = GT_META_CALL(
            meta::flatten, (GT_META_CALL(meta::transform, (meta::third, loop_intervals_t))));
        using all_stages_t = GT_META_CALL(meta::flatten, all_stage_groups_t);

        template <class Stage>
        GT_META_DEFINE_ALIAS(get_stage_extent, meta::id, typename Stage::extent_t);

        using all_extents_t = GT_META_CALL(meta::transform, (get_stage_extent, all_stages_t));

      public:
        using max_extent_t = GT_META_CALL(meta::rename, (enclosing_extent, all_extents_t));

        typedef LocalDomain local_domain_t;
        typedef CacheSequence cache_sequence_t;
        typedef Grid grid_t;
        typedef ExecutionEngine execution_type_t;
    };

    template <class T>
    GT_META_DEFINE_ALIAS(is_run_functor_arguments, meta::is_instantiation_of, (run_functor_arguments, T));
} // namespace gridtools
