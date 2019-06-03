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

#include "combine.hpp"
#include "concat.hpp"
#include "macros.hpp"

namespace gridtools {
    /**
     *  Flatten a list of lists.
     *
     *  Note: this function doesn't go recursive. It just concatenates the inner lists.
     */
    namespace meta {
        namespace lazy {
            template <class Lists>
            using flatten = combine<meta::concat, Lists>;
        }
        template <class Lists>
        using flatten = typename lazy::combine<concat, Lists>::type;
    } // namespace meta
} // namespace gridtools
