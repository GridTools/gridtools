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

namespace gridtools {
    namespace meta {
        namespace internal {
            template <class... Ts>
            struct inherit : Ts... {};
        } // namespace internal
    }     // namespace meta
} // namespace gridtools
