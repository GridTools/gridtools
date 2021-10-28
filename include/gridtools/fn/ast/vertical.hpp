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

#include <type_traits>

#include "generate.hpp"
#include "nodes.hpp"
#include "parse.hpp"

namespace gridtools::fn::ast {
    namespace vertical_impl_ {
        template <class>
        struct has_scans : std::false_type {};

        template <template <class...> class Node, class... Trees>
        struct has_scans<Node<Trees...>> : std::disjunction<has_scans<Trees>...> {};

        template <auto F, class... Trees>
        struct has_scans<lambda<meta::val<F>, Trees...>> : has_scans<decltype(F(Trees()...))> {};

        template <class F, class... Trees>
        struct has_scans<inlined<F, Trees...>> : has_scans<lambda<F, Trees...>> {};

        template <class F, class... Trees>
        struct has_scans<tmp<F, Trees...>> : has_scans<lambda<F, Trees...>> {};

        template <class... Ts>
        struct has_scans<builtin<builtins::scan, Ts...>> : std::true_type {};
    } // namespace vertical_impl_
    using vertical_impl_::has_scans;
} // namespace gridtools::fn::ast