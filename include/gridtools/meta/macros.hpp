/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <type_traits>

#include <boost/preprocessor.hpp>

// internal
#define GT_META_INTERNAL_APPLY(fun, args) BOOST_PP_REMOVE_PARENS(fun)<BOOST_PP_REMOVE_PARENS(args)>

#define GT_META_DELEGATE_TO_LAZY(fun, signature, args) \
    template <BOOST_PP_REMOVE_PARENS(signature)>       \
    using fun = typename lazy::fun<BOOST_PP_REMOVE_PARENS(args)>::type

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
