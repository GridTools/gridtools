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
#include "../arith.hpp"
#include "../deref.hpp"
#include "../lambda.hpp"
#include "../lift.hpp"
#include "../shift.hpp"
#include "../tuple.hpp"
#include "nodes.hpp"

namespace gridtools::fn::ast {
    template <class Tree>
    constexpr auto generate = generator(Tree());

    template <auto F, auto... Fs>
    constexpr auto apply = [](auto const &... args) -> decltype(auto) { return F(Fs(args...)...); };

    template <class V>
    consteval auto generator(constant<V>) {
        return [](...) -> V { return {}; };
    }

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

    template <class... Args>
    consteval auto generator(plus<Args...>) {
        return apply<fn::plus, generate<Args>...>;
    }

    template <class... Args>
    consteval auto generator(minus<Args...>) {
        return apply<fn::minus, generate<Args>...>;
    }

    template <class... Args>
    consteval auto generator(multiplies<Args...>) {
        return apply<fn::multiplies, generate<Args>...>;
    }

    template <class... Args>
    consteval auto generator(divides<Args...>) {
        return apply<fn::divides, generate<Args>...>;
    }

    template <class... Args>
    consteval auto generator(make_tuple<Args...>) {
        return apply<fn::make_tuple, generate<Args>...>;
    }

    template <size_t I, class... Args>
    consteval auto generator(tuple_get<meta::val<I>, Args...>) {
        return apply<fn::tuple_get<I>, generate<Args>...>;
    }

    template <class Tree>
    consteval auto generator(deref<Tree>) {
        return apply<fn::deref, generate<Tree>>;
    }

    template <class Tree, auto... Offsets>
    consteval auto generator(shifted<Tree, meta::val<Offsets...>>) {
        return apply<fn::shift<Offsets...>, generate<Tree>>;
    }

    template <auto F, class... Args>
    consteval auto generator(inlined<meta::val<F>, Args...>) {
        return apply<fn::ilift<F>, generate<Args>...>;
    };

    template <auto F, class... Args>
    consteval auto generator(tmp<meta::val<F>, Args...>) {
        return apply<fn::tlift<F>, generate<Args>...>;
    };
} // namespace gridtools::fn::ast
