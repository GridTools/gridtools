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

#include "../../meta.hpp"
#include "cache_info.hpp"
#include "esf.hpp"
#include "execution_types.hpp"

/**
@file
@brief descriptor of the Multi Stage Stencil (MSS)
*/
namespace gridtools {
    namespace stencil {
        namespace core {
            /** @brief Descriptors for  Multi Stage Stencil (MSS) */
            template <class ExecutionEngine, class EsfDescrSequence, class CacheMap>
            struct mss_descriptor {
                static_assert(is_execution_engine<ExecutionEngine>::value, GT_INTERNAL_ERROR);
                static_assert(meta::all_of<is_esf_descriptor, EsfDescrSequence>::value, GT_INTERNAL_ERROR);

                using execution_engine_t = ExecutionEngine;
                using esf_sequence_t = EsfDescrSequence;
                using cache_map_t = CacheMap;
            };

            template <typename Mss>
            using is_mss_descriptor = meta::is_instantiation_of<mss_descriptor, Mss>;
        } // namespace core
    }     // namespace stencil
} // namespace gridtools
