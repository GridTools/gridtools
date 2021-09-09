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

#include <utility>

namespace gridtools::fn {
    namespace scan_impl_ {
        constexpr auto scan(auto fun, auto init, bool is_backward = false) {
            return [fun = std::move(fun), init = std::move(init)](auto const &... args) {

            }
        }
    } // namespace scan_impl_
    using scan_impl_::scan;
} // namespace gridtools::fn
