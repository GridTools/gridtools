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

#include <cstddef>

#include "concat.hpp"
#include "drop_front.hpp"
#include "macros.hpp"
#include "push_front.hpp"
#include "take.hpp"

namespace gridtools {
    namespace meta {
        template <size_t N, class List, class... Ts>
        GT_META_DEFINE_ALIAS(insert_c,
            concat,
            (GT_META_CALL(take_c, (N, List)),
                GT_META_CALL(push_front, (GT_META_CALL(drop_front_c, (N, List)), Ts...))));

        template <class N, class List, class... Ts>
        GT_META_DEFINE_ALIAS(insert,
            concat,
            (GT_META_CALL(take_c, (N::value, List)),
                GT_META_CALL(push_front, (GT_META_CALL(drop_front_c, (N::value, List)), Ts...))));
    } // namespace meta
} // namespace gridtools
