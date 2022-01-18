/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../common/hymap.hpp"
#include "../common/integral_constant.hpp"

namespace gridtools::fn {

    template <class Stencil, class MakeIterator, int Out, int... Ins>
    struct stencil_stage {
        template <class Ptr, class Strides>
        GT_FUNCTION void operator()(Ptr &ptr, Strides const &strides) const {
            *at_key<integral_constant<int, Out>>(ptr) =
                Stencil()()(MakeIterator()()(integral_constant<int, Ins>(), ptr, strides)...);
        }
    };

    template <class... Stages>
    struct merged_stencil_stage {
        template <class Ptr, class Strides>
        GT_FUNCTION void operator()(Ptr &ptr, Strides const &strides) const {
            (Stages()(ptr, strides), ...);
        }
    };

} // namespace gridtools::fn
