/*
  GridTools Libraries

  Copyright (c) 2017, ETH Zurich and MeteoSwiss
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  1. Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

  For information: http://eth-cscs.github.io/gridtools/
*/

#pragma once

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

#define GT_META_LAZY_NAMESPASE inline namespace lazy

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

#define GT_META_LAZY_NAMESPASE namespace lazy

#define GT_META_DELEGATE_TO_LAZY(fun, signature, args) \
    template <BOOST_PP_REMOVE_PARENS(signature)>       \
    using fun = typename lazy::fun<BOOST_PP_REMOVE_PARENS(args)>::type

#endif
