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

#include "../common/defs.hpp"
#include "../meta/is_instantiation_of.hpp"
#include "../meta/logical.hpp"
#include "caches/cache_traits.hpp"
#include "color.hpp"
#include "esf_fwd.hpp"
#include "extent.hpp"
#include "local_domain.hpp"
#include "loop_interval.hpp"

namespace gridtools {

    template <typename Backend, typename LocalDomain, typename EsfSequence>
    struct iterate_domain_arguments {
        GT_STATIC_ASSERT(is_local_domain<LocalDomain>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_esf_descriptor, EsfSequence>::value), GT_INTERNAL_ERROR);

        typedef Backend backend_t;
        typedef LocalDomain local_domain_t;
        typedef EsfSequence esf_sequence_t;
    };

    template <class T>
    using is_iterate_domain_arguments = meta::is_instantiation_of<iterate_domain_arguments, T>;

    /**
     * @brief type that contains main metadata required to execute a mss kernel. This type will be passed to
     * all functors involved in the execution of the mss
     */
    template <typename Backend,  // id of the different backends
        typename EsfSequence,    // sequence of ESF
        typename LoopIntervals,  // loop intervals
        typename ExecutionEngine // the execution engine
        >
    struct run_functor_arguments {
      private:
        GT_STATIC_ASSERT(is_execution_engine<ExecutionEngine>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_esf_descriptor, EsfSequence>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_loop_interval, LoopIntervals>::value), GT_INTERNAL_ERROR);

        using all_stage_groups_t = meta::flatten<meta::transform<meta::third, LoopIntervals>>;
        using all_stages_t = meta::flatten<all_stage_groups_t>;

        template <class Stage>
        using get_stage_extent = typename Stage::extent_t;

        using all_extents_t = meta::transform<get_stage_extent, all_stages_t>;

      public:
        using backend_t = Backend;

        // needed for:
        // 1. compute_readwrite_args
        // 2. get_k_cache_storage_tuple (extract_k_extent_for_cache)
        using esf_sequence_t = EsfSequence;
        using loop_intervals_t = LoopIntervals;
        using execution_type_t = ExecutionEngine;
        using max_extent_t = meta::rename<enclosing_extent, all_extents_t>;
    };

    template <class T>
    using is_run_functor_arguments = meta::is_instantiation_of<run_functor_arguments, T>;
} // namespace gridtools
