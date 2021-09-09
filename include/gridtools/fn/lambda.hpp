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

#include "../meta.hpp"

namespace gridtools::fn {
    namespace lambda_impl_ {
        struct undefined {};
        undefined fn_lambda(...) { return {}; }

        template <auto F>
        constexpr auto lambda = []<class... Args>(Args &&... args) {
            if constexpr (std::is_same_v<decltype(fn_lambda(meta::constant<F>, std::forward<Args>(args)...)),
                              undefined>)
                return F(std::forward<Args>(args)...);
            else
                return fn_lambda(meta::constant<F>, std::forward<Args>(args)...);
        };
    } // namespace lambda_impl_
    using lambda_impl_::lambda;
} // namespace gridtools::fn
