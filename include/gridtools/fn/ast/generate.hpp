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

#include <tuple>

#include "../../meta.hpp"
#include "../builtins.hpp"
#include "../lambda.hpp"
#include "nodes.hpp"

namespace gridtools::fn::ast {
    template <auto F, auto... Fs>
    constexpr auto apply = [](auto const &... args) -> decltype(auto) { return F(Fs(args...)...); };

    template <class V>
    consteval auto generator(V) {
        return [](...) { return V(); };
    }

    template <class Tree>
    constexpr auto generate = generator(Tree());

    template <class I>
    consteval auto generator(in<I>) {
        return []<class... Args>(Args && ... args)->decltype(auto) {
            return std::get<I::value>(std::forward_as_tuple(std::forward<Args>(args)...));
        };
    }

    template <auto F, class... Args>
    consteval auto generator(lambda<meta::val<F>, Args...>) {
        return apply<fn::lambda<F>, generate<Args>...>;
    }

    template <class Tag, class... Args>
    consteval auto generator(builtin<Tag, Args...>) {
        return apply<fn::builtin<Tag>, generate<Args>...>;
    }
} // namespace gridtools::fn::ast
