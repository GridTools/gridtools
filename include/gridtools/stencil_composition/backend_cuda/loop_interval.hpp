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

#include "../../common/defs.hpp"
#include "../../common/generic_metafunctions/for_each.hpp"
#include "../../common/host_device.hpp"

namespace gridtools {
    namespace cuda {
        template <class Count, class Stages>
        class loop_interval {
            Count m_count;

          public:
            using count_t = Count;
            using stages_t = Stages;

            loop_interval(Count count) : m_count(count) {}

            static Stages stages() { return {}; }

            GT_FUNCTION Count count() const { return m_count; }

            template <class Ptr, class Strides, class Validator>
            GT_FUNCTION_DEVICE void operator()(
                Ptr const &GT_RESTRICT ptr, Strides const &GT_RESTRICT strides, Validator const &validator) const {
                device::for_each<Stages>([&](auto stage) GT_FORCE_INLINE_LAMBDA { stage(ptr, strides, validator); });
            }
        };

        template <class Count, class Stages>
        loop_interval<Count, Stages> make_loop_interval(Count count, Stages) {
            return {count};
        }
    } // namespace cuda
} // namespace gridtools
