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

#include "../meta.hpp"
#include "compute_extents_metafunctions.hpp"
#include "esf_metafunctions.hpp"
#include "make_loop_intervals.hpp"
#include "mss.hpp"
#include "stages_maker.hpp"

namespace gridtools {
    /**
     * @brief the mss components contains meta data associated to a mss descriptor.
     * All derived metadata is computed in this class
     * @tparam MssDescriptor the mss descriptor
     * @tparam ExtentSizes the extent sizes of all the ESFs in this mss
     */
    template <typename MssDescriptor, typename ExtentMap, typename Axis>
    struct mss_components {
        GT_STATIC_ASSERT(is_mss_descriptor<MssDescriptor>::value, GT_INTERNAL_ERROR);
        typedef MssDescriptor mss_descriptor_t;

        typedef typename MssDescriptor::execution_engine_t execution_engine_t;

        // calculate loop intervals and order them according to the execution policy.
        using loop_intervals_t = order_loop_intervals<execution_engine_t,
            make_loop_intervals<stages_maker<MssDescriptor, ExtentMap>::template apply, Axis>>;
    };
} // namespace gridtools
