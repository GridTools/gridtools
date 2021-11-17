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

        template <class... Ts, class... Us>
        struct has_scans<builtin<builtins::scan<Ts...>, Us...>> : std::true_type {};

        template <class, class = void>
        struct is_scan : std::false_type {};

        template <class IsBackward, class Init, class Body, class Prologues, class Epilogues, class... Is>
        struct is_scan<builtin<builtins::scan<IsBackward, Init, Body, Prologues, Epilogues>, in<Is>...>,
            std::enable_if_t<std::is_same_v<meta::list<Is...>, meta::make_indices_c<sizeof...(Is)>>>> : std::true_type {
        };
    } // namespace vertical_impl_
    using vertical_impl_::has_scans;
    using vertical_impl_::is_scan;
} // namespace gridtools::fn::ast