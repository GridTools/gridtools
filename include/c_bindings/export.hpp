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

#include <boost/function_types/function_arity.hpp>
#include <boost/function_types/result_type.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/mpl/at.hpp>
#include <boost/preprocessor.hpp>

#include "function_wrapper.hpp"
#include "generator.hpp"

#define GT_EXPORT_BINDING_IMPL_PARAM_DECL(z, i, signature)                                                       \
    typename boost::mpl::at_c<                                                                                   \
        typename boost::function_types::parameter_types<::gridtools::c_bindings::wrapped_t< signature > >::type, \
        i >::type param_##i

/**
 *   Defines the function with the given name with the C linkage.
 *
 *   The signature if the generated function will be transformed from the provided signature according to the following
 *   rules:
 *     - for result type:
 *       - `void` and arithmetic types remain the same;
 *       - classes (and structures) and references to them are transformed to the pointer to the opaque handle
 *         (`gt_handle*`) which should be released by calling `void gt_release(gt_handle*)` function;
 *       - all other result types will cause a compiler error.
 *     - for parameter types:
 *       - arithmetic types and pointers to them remain the same;
 *       - references to arithmetic types are transformed to the corresponded pointers;
 *       - classes (and structures) and references or pointers to them are transformed to `gt_handle*`;
 *       - all other parameter types will cause a compiler error.
 *   Additionally the newly generated function will be registered for automatic interface generation.
 *
 *   @param n The arity of the generated function.
 *   @param name The name of the generated function.
 *   @param signature The signature that will be used to invoke `impl`.
 *   @param impl The functor that the generated function will delegate to.
 */
#define GT_EXPORT_BINDING_WITH_SIGNATURE(n, name, signature, impl)                                                  \
    static_assert(::boost::function_types::function_arity< signature >() == n, "arity mismatch");                   \
    extern "C"                                                                                                      \
        typename ::boost::function_types::result_type<::gridtools::c_bindings::wrapped_t< signature > >::type name( \
            BOOST_PP_ENUM(n, GT_EXPORT_BINDING_IMPL_PARAM_DECL, signature)) {                                       \
        return ::gridtools::c_bindings::wrap< signature >(impl)(BOOST_PP_ENUM_PARAMS(n, param_));                   \
    }                                                                                                               \
    GT_ADD_GENERATED_DECLARATION(::gridtools::c_bindings::wrapped_t< signature >, name)

/// The flavour of GT_EXPORT_BINDING_WITH_SIGNATURE where the `impl` parameter is a function pointer.
#define GT_EXPORT_BINDING(n, name, impl) GT_EXPORT_BINDING_WITH_SIGNATURE(n, name, decltype(impl), impl)

/// GT_EXPORT_BINDING_WITH_SIGNATURE shortcuts for the given arity
#define GT_EXPORT_BINDING_WITH_SIGNATURE_0(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE(0, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_1(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE(1, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_2(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE(2, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_3(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE(3, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_4(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE(4, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_5(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE(5, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_6(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE(6, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_7(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE(7, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_8(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE(8, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_9(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE(9, name, s, i)

/// GT_EXPORT_BINDING shortcuts for the given arity
#define GT_EXPORT_BINDING_0(name, impl) GT_EXPORT_BINDING(0, name, impl)
#define GT_EXPORT_BINDING_1(name, impl) GT_EXPORT_BINDING(1, name, impl)
#define GT_EXPORT_BINDING_2(name, impl) GT_EXPORT_BINDING(2, name, impl)
#define GT_EXPORT_BINDING_3(name, impl) GT_EXPORT_BINDING(3, name, impl)
#define GT_EXPORT_BINDING_4(name, impl) GT_EXPORT_BINDING(4, name, impl)
#define GT_EXPORT_BINDING_5(name, impl) GT_EXPORT_BINDING(5, name, impl)
#define GT_EXPORT_BINDING_6(name, impl) GT_EXPORT_BINDING(6, name, impl)
#define GT_EXPORT_BINDING_7(name, impl) GT_EXPORT_BINDING(7, name, impl)
#define GT_EXPORT_BINDING_8(name, impl) GT_EXPORT_BINDING(8, name, impl)
#define GT_EXPORT_BINDING_9(name, impl) GT_EXPORT_BINDING(9, name, impl)
