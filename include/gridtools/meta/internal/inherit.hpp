/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
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
