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

#include "../../common/integral_constant.hpp"
#include "../../meta.hpp"
#include "nodes.hpp"

namespace gridtools::fn::ast {
    namespace parse_impl_ {
        template <auto F, class>
        struct parse;

        template <auto F, class... Is>
        struct parse<F, meta::list<Is...>> {
            using type = decltype(F(in<Is>()...));
        };
    } // namespace parse_impl_

    template <auto F, class... Args>
    using parse = typename parse_impl_::parse<F, meta::make_indices_c<sizeof...(Args)>>::type;

    template <class F, class... Args>
    lambda<F, Args...> fn_lambda(F, Args const &...) {
        return {};
    }

    template <class Tag, class... Args>
    builtin<Tag, Args...> fn_builtin(Tag, Args...) {
        return {};
    }
} // namespace gridtools::fn::ast
