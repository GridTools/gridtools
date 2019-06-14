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
#include "../meta/macros.hpp"
#include "caches/cache_traits.hpp"
#include "esf.hpp"
#include "execution_types.hpp"

/**
@file
@brief descriptor of the Multi Stage Stencil (MSS)
*/
namespace gridtools {
    /** @brief Descriptors for  Multi Stage Stencil (MSS) */
    template <typename ExecutionEngine, typename EsfDescrSequence, typename CacheSequence>
    struct mss_descriptor {
        GT_STATIC_ASSERT((meta::all_of<is_esf_descriptor, EsfDescrSequence>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT((meta::all_of<is_cache, CacheSequence>::value), GT_INTERNAL_ERROR);
        GT_STATIC_ASSERT(is_execution_engine<ExecutionEngine>::value, GT_INTERNAL_ERROR);

        using execution_engine_t = ExecutionEngine;
        using esf_sequence_t = EsfDescrSequence;
        using cache_sequence_t = CacheSequence;
    };

    template <typename Mss>
    using is_mss_descriptor = meta::is_instantiation_of<mss_descriptor, Mss>;
} // namespace gridtools
