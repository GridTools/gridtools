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

        namespace lazy {
            template <class T>
            struct normalize {
                using type = T;
            };

            template <class T, T V>
            struct normalize<integral_constant<T, V>> {
                using type = constant<integral_constant<T, V>>;
            };
        } // namespace lazy
        GT_META_DELEGATE_TO_LAZY(normalize, class T, T);
    } // namespace parse_impl_

    template <auto F, class... Args>
    using parse = typename parse_impl_::parse<F, meta::make_indices_c<sizeof...(Args)>>::type;

    template <class F, class... Args>
    lambda<F, parse_impl_::normalize<Args>...> fn_lambda(F, Args const &...) {
        return {};
    }

    template <class... Args>
    plus<parse_impl_::normalize<Args>...> fn_plus(Args const &...) {
        return {};
    }

    template <class... Args>
    minus<parse_impl_::normalize<Args>...> fn_minus(Args const &...) {
        return {};
    }

    template <class... Args>
    multiplies<parse_impl_::normalize<Args>...> fn_multiplies(Args const &...) {
        return {};
    }

    template <class... Args>
    divides<parse_impl_::normalize<Args>...> fn_divides(Args const &...) {
        return {};
    }

    template <class... Args>
    make_tuple<parse_impl_::normalize<Args>...> fn_make_tuple(Args const &...) {
        return {};
    }

    template <class I, class... Args>
    tuple_get<I, parse_impl_::normalize<Args>...> fn_tuple_get(I, Args const &...) {
        return {};
    }

    template <class Tree>
    deref<Tree> fn_deref(Tree const &) {
        return {};
    }

    template <class Tree, class Offsets>
    shifted<Tree, Offsets> fn_shift(Tree const &, Offsets) {
        return {};
    }

    template <class F, class... Args>
    inlined<F, Args...> fn_ilift(F, Args const &...) {
        return {};
    }

    template <class F, class... Args>
    tmp<F, Args...> fn_tlift(F, Args const &...) {
        return {};
    }
} // namespace gridtools::fn::ast
