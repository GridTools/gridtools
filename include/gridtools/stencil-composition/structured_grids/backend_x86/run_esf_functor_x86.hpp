/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../../common/defs.hpp"
#include "../../../common/host_device.hpp"
#include "../../../meta.hpp"

namespace gridtools {
    struct run_esf_functor_x86 {
        template <class StageGroups, class ItDomain>
        GT_FUNCTION static void exec(ItDomain &it_domain) {
            using stages_t = GT_META_CALL(meta::flatten, StageGroups);
            GT_STATIC_ASSERT(meta::length<stages_t>::value == 1, GT_INTERNAL_ERROR);
            using stage_t = GT_META_CALL(meta::first, stages_t);
            stage_t::exec(it_domain);
        }
    };
} // namespace gridtools
