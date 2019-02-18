/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <type_traits>

#include <boost/preprocessor.hpp>

#include "defs.hpp"

// internal
#define GT_META_INTERNAL_APPLY(fun, args) BOOST_PP_REMOVE_PARENS(fun)<BOOST_PP_REMOVE_PARENS(args)>

#if GT_BROKEN_TEMPLATE_ALIASES

/**
 * backward compatible way to call function
 */
#define GT_META_CALL(fun, args) typename GT_META_INTERNAL_APPLY(fun, args)::type

/**
 * backward compatible way to define an alias to the function composition
 */
#define GT_META_DEFINE_ALIAS(name, fun, args) \
    struct name : GT_META_INTERNAL_APPLY(fun, args) {}

#define GT_META_LAZY_NAMESPACE inline namespace lazy

#define GT_META_DELEGATE_TO_LAZY(fun, signature, args) static_assert(1, "")

#else

/**
 * backward compatible way to call function
 */
#define GT_META_CALL(fun, args) GT_META_INTERNAL_APPLY(fun, args)
/**
 * backward compatible way to define an alias to the function composition
 */
#define GT_META_DEFINE_ALIAS(name, fun, args) using name = GT_META_INTERNAL_APPLY(fun, args)

#define GT_META_LAZY_NAMESPACE namespace lazy

#define GT_META_DELEGATE_TO_LAZY(fun, signature, args) \
    template <BOOST_PP_REMOVE_PARENS(signature)>       \
    using fun = typename lazy::fun<BOOST_PP_REMOVE_PARENS(args)>::type

#endif

/**
 *  NVCC bug workaround: sizeof... works incorrectly within template alias context.
 */
#ifdef __NVCC__

namespace gridtools {
    namespace meta {
        template <class... Ts>
        struct sizeof_3_dots : std::integral_constant<std::size_t, sizeof...(Ts)> {};
    } // namespace meta
} // namespace gridtools

#define GT_SIZEOF_3_DOTS(Ts) ::gridtools::meta::sizeof_3_dots<Ts...>::value
#else
#define GT_SIZEOF_3_DOTS(Ts) sizeof...(Ts)
#endif
