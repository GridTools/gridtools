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

#include <boost/function_types/function_arity.hpp>
#include <boost/function_types/parameter_types.hpp>
#include <boost/function_types/result_type.hpp>
#include <boost/mpl/at.hpp>
#include <boost/preprocessor.hpp>

#include "../common/boost_pp_generic_macros.hpp"
#include "function_wrapper.hpp"
#include "generator.hpp"

#define GT_EXPORT_BINDING_IMPL_PARAM_DECL(z, i, signature)                                                    \
    typename boost::mpl::at_c<                                                                                \
        typename boost::function_types::parameter_types<::gridtools::c_bindings::wrapped_t<signature>>::type, \
        i>::type param_##i

#define GT_ADD_GENERATED_DEFINITION_IMPL(n, name, cppsignature, impl)                                                \
    static_assert(::boost::function_types::function_arity<cppsignature>::value == n, "arity mismatch");              \
    extern "C" typename ::boost::function_types::result_type<::gridtools::c_bindings::wrapped_t<cppsignature>>::type \
    name(BOOST_PP_ENUM(n, GT_EXPORT_BINDING_IMPL_PARAM_DECL, cppsignature)) {                                        \
        return ::gridtools::c_bindings::wrap<cppsignature>(impl)(BOOST_PP_ENUM_PARAMS(n, param_));                   \
    }

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
 *       - types that fulfill the concept of being fortran_array_bindable are transformed to a
 *         gt_fortran_array_descriptor
 *       - classes (and structures) and references or pointers to them are transformed to `gt_handle*`;
 *       - all other parameter types will cause a compiler error.
 *   Additionally the newly generated function will be registered for automatic interface generation.
 *
 *   @param n The arity of the generated function.
 *   @param name The name of the generated function.
 *   @param cppsignature The signature that will be used to invoke `impl`.
 *   @param impl The functor that the generated function will delegate to.
 */
#define GT_EXPORT_BINDING_WITH_SIGNATURE(n, name, cppsignature, impl) \
    GT_ADD_GENERATED_DEFINITION_IMPL(n, name, cppsignature, impl)     \
    GT_ADD_GENERATED_DECLARATION(::gridtools::c_bindings::wrapped_t<cppsignature>, name)

/**
 *   Defines the function with the given name with the C linkage with an additional wrapper in the fortran bindings. The
 *   additional wrapper provides the following functionality:
 *     - It generates a gt_fortran_array_descriptor from an array, if the type on C++-side is fortran_array_wrappable.
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
 *       - types that are fortran_array_bindable but not fortran_array_wrappable are transformed to a
 *         gt_fortran_array_descriptor
 *       - types that are fortran_array_wrappable are transformed to a gt_fortran_array_descriptor in the c-bindings,
 *         and provide a wrapper in the fortran-bindings such that they can be called with a fortran array
 *       - classes (and structures) and references or pointers to them are transformed to `gt_handle*`;
 *       - all other parameter types will cause a compiler error.
 *   Additionally the newly generated function will be registered for automatic interface generation.
 *
 *   @param n The arity of the generated function.
 *   @param name The name of the generated function.
 *   @param cppsignature The signature that will be used to invoke `impl`.
 *   @param impl The functor that the generated function will delegate to.
 */
#define GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED(n, name, cppsignature, impl) \
    GT_ADD_GENERATED_DEFINITION_IMPL(n, name, cppsignature, impl)             \
    GT_ADD_GENERATED_DECLARATION_WRAPPED(cppsignature, name)

/// The flavour of GT_EXPORT_BINDING_WITH_SIGNATURE where the `impl` parameter is a function pointer.
#define GT_EXPORT_BINDING(n, name, impl) \
    GT_EXPORT_BINDING_WITH_SIGNATURE(n, name, decltype(BOOST_PP_REMOVE_PARENS(impl)), impl)
#define GT_EXPORT_BINDING_WRAPPED(n, name, impl) \
    GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED(n, name, decltype(BOOST_PP_REMOVE_PARENS(impl)), impl)

#define GT_EXPORT_GENERIC_BINDING_IMPL_IMPL(generatorsuffix, n, generic_name, concrete_name, impl) \
    BOOST_PP_CAT(GT_EXPORT_BINDING, generatorsuffix)(n, concrete_name, impl);                      \
    GT_ADD_GENERIC_DECLARATION(generic_name, concrete_name)

#define GT_EXPORT_GENERIC_BINDING_IMPL(generatorsuffix, n, name, suffix, impl) \
    GT_EXPORT_GENERIC_BINDING_IMPL_IMPL(generatorsuffix, n, name, BOOST_PP_CAT(name, suffix), impl)

#define GT_EXPORT_GENERIC_BINDING_IMPL_FUNCTOR(r, data, i, elem)    \
    GT_EXPORT_GENERIC_BINDING_IMPL(BOOST_PP_TUPLE_ELEM(4, 0, data), \
        BOOST_PP_TUPLE_ELEM(4, 1, data),                            \
        BOOST_PP_TUPLE_ELEM(4, 2, data),                            \
        i,                                                          \
        (BOOST_PP_TUPLE_ELEM(4, 3, data) < BOOST_PP_REMOVE_PARENS(elem) >));

#define GT_EXPORT_GENERIC_BINDING(n, name, impl_template, template_params) \
    BOOST_PP_SEQ_FOR_EACH_I(GT_EXPORT_GENERIC_BINDING_IMPL_FUNCTOR,        \
        (, n, name, impl_template),                                        \
        BOOST_PP_VARIADIC_SEQ_TO_SEQ(template_params))                     \
    static_assert(1, "")

#define GT_EXPORT_GENERIC_BINDING_WRAPPED(n, name, impl_template, template_params) \
    BOOST_PP_SEQ_FOR_EACH_I(GT_EXPORT_GENERIC_BINDING_IMPL_FUNCTOR,                \
        (_WRAPPED, n, name, impl_template),                                        \
        BOOST_PP_VARIADIC_SEQ_TO_SEQ(template_params))                             \
    static_assert(1, "")

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

#define GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED_0(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED(0, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED_1(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED(1, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED_2(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED(2, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED_3(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED(3, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED_4(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED(4, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED_5(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED(5, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED_6(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED(6, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED_7(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED(7, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED_8(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED(8, name, s, i)
#define GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED_9(name, s, i) GT_EXPORT_BINDING_WITH_SIGNATURE_WRAPPED(9, name, s, i)

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

#define GT_EXPORT_BINDING_WRAPPED_0(name, impl) GT_EXPORT_BINDING_WRAPPED(0, name, impl)
#define GT_EXPORT_BINDING_WRAPPED_1(name, impl) GT_EXPORT_BINDING_WRAPPED(1, name, impl)
#define GT_EXPORT_BINDING_WRAPPED_2(name, impl) GT_EXPORT_BINDING_WRAPPED(2, name, impl)
#define GT_EXPORT_BINDING_WRAPPED_3(name, impl) GT_EXPORT_BINDING_WRAPPED(3, name, impl)
#define GT_EXPORT_BINDING_WRAPPED_4(name, impl) GT_EXPORT_BINDING_WRAPPED(4, name, impl)
#define GT_EXPORT_BINDING_WRAPPED_5(name, impl) GT_EXPORT_BINDING_WRAPPED(5, name, impl)
#define GT_EXPORT_BINDING_WRAPPED_6(name, impl) GT_EXPORT_BINDING_WRAPPED(6, name, impl)
#define GT_EXPORT_BINDING_WRAPPED_7(name, impl) GT_EXPORT_BINDING_WRAPPED(7, name, impl)
#define GT_EXPORT_BINDING_WRAPPED_8(name, impl) GT_EXPORT_BINDING_WRAPPED(8, name, impl)
#define GT_EXPORT_BINDING_WRAPPED_9(name, impl) GT_EXPORT_BINDING_WRAPPED(9, name, impl)
