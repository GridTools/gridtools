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

#include <boost/preprocessor.hpp>

#define GT_DEFINE_ACCESSORS_IMPL_NAME_FROM_DEF_OP(seq, data, accessor_def) BOOST_PP_SEQ_ELEM(1, accessor_def)

#define GT_DEFINE_ACCESSORS_IMPL_TYPEDEF_OP(r, data, i, accessor_def) \
    using BOOST_PP_SEQ_ELEM(1, accessor_def) = BOOST_PP_SEQ_ELEM(     \
        0, accessor_def)<BOOST_PP_SEQ_ENUM(BOOST_PP_SEQ_REPLACE(BOOST_PP_SEQ_TAIL(accessor_def), 0, i))>;

#define GT_DEFINE_ACCESSORS_IMPL_DEFINE_PARAM_LIST(accessor_def_sec)                  \
    BOOST_PP_SEQ_FOR_EACH_I(GT_DEFINE_ACCESSORS_IMPL_TYPEDEF_OP, 0, accessor_def_sec) \
    using param_list = make_param_list<BOOST_PP_SEQ_ENUM(                             \
        BOOST_PP_SEQ_TRANSFORM(GT_DEFINE_ACCESSORS_IMPL_NAME_FROM_DEF_OP, 0, accessor_def_sec))>

#define GT_DEFINE_ACCESSORS_IMPL(accessor_def_sec) GT_DEFINE_ACCESSORS_IMPL_DEFINE_PARAM_LIST(accessor_def_sec)

/**
 *  Micro EDSL for defining accessors within stencil functors
 *
 *   Usage example:
 *
 *   struct my_fun {
 *       GT_DEFINE_ACCESSORS(
 *         GT_INOUT_ACCESSOR(out),
 *         GT_IN_ACCESSOR(in, extent<-1, 1>)
 *       );
 *
 *       template <class Eval>
 *       static GT_FUNCTION void apply(Eval eval) {
 *          eval(out()) = (eval(in(-1)) + eval(in(1))) / 2;
 *       }
 *   };
 *
 */

#define GT_INOUT_ACCESSOR(...) BOOST_PP_VARIADIC_TO_SEQ(::gridtools::inout_accessor, __VA_ARGS__)
#define GT_IN_ACCESSOR(...) BOOST_PP_VARIADIC_TO_SEQ(::gridtools::in_accessor, __VA_ARGS__)
#define GT_INOUT_ACCESSOR(...) BOOST_PP_VARIADIC_TO_SEQ(::gridtools::inout_accessor, __VA_ARGS__)
#define GT_GLOBAL_ACCESSOR(name) (::gridtools::global_accessor)(name)

#define GT_DEFINE_ACCESSORS(...) GT_DEFINE_ACCESSORS_IMPL(BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
