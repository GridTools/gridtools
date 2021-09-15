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

#include "../../meta.hpp"
#include "../builtins.hpp"

namespace gridtools::fn::ast {
    template <class...>
    struct in;

    template <class I>
    struct in<I> {};

    template <class...>
    struct lambda;

    template <auto F, class... Args>
    struct lambda<meta::val<F>, Args...> {};

    template <class...>
    struct builtin;

    template <class Tag, class... Args>
    struct builtin<Tag, Args...> {};

    template <class F, class... Trees>
    using inlined = builtin<builtins::ilift, F, Trees...>;

    template <class F, class... Trees>
    using tmp = builtin<builtins::tlift, F, Trees...>;

    template <class T>
    using deref = builtin<builtins::deref, T>;

    template <class Tree, class Offsets>
    using shifted = builtin<builtins::shift, Offsets, Tree>;

    template <class T>
    using deref = builtin<builtins::deref, T>;
} // namespace gridtools::fn::ast
