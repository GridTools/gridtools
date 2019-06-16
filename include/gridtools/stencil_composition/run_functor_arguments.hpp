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
#include "../meta.hpp"
#include "execution_types.hpp"
#include "loop_interval.hpp"

namespace gridtools {
    /**
     * @brief type that contains main metadata required to execute a mss kernel. This type will be passed to
     * all functors involved in the execution of the mss
     */
    template <typename LoopIntervals, typename ExecutionEngine>
    struct run_functor_arguments {
        GT_STATIC_ASSERT(is_execution_engine<ExecutionEngine>::value, GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_loop_interval, LoopIntervals>::value), GT_INTERNAL_ERROR);

        using loop_intervals_t = LoopIntervals;
        using execution_type_t = ExecutionEngine;
    };

    template <class T>
    using is_run_functor_arguments = meta::is_instantiation_of<run_functor_arguments, T>;
} // namespace gridtools
