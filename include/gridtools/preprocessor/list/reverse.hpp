# /* Copyright (C) 2001
#  * Housemarque Oy
#  * http://www.housemarque.com
#  *
#  * Distributed under the Boost Software License, Version 1.0. (See
#  * accompanying file LICENSE_1_0.txt or copy at
#  * http://www.boost.org/LICENSE_1_0.txt)
#  */
#
# /* Revised by Paul Mensonides (2002) */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_LIST_REVERSE_HPP
# define GT_PREPROCESSOR_LIST_REVERSE_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_STRICT()
#
# include <gridtools/preprocessor/list/fold_left.hpp>
#
# /* GT_PP_LIST_REVERSE */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_LIST_REVERSE(list) GT_PP_LIST_FOLD_LEFT(GT_PP_LIST_REVERSE_O, GT_PP_NIL, list)
# else
#    define GT_PP_LIST_REVERSE(list) GT_PP_LIST_REVERSE_I(list)
#    define GT_PP_LIST_REVERSE_I(list) GT_PP_LIST_FOLD_LEFT(GT_PP_LIST_REVERSE_O, GT_PP_NIL, list)
# endif
#
# define GT_PP_LIST_REVERSE_O(d, s, x) (x, s)
#
# /* GT_PP_LIST_REVERSE_D */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_LIST_REVERSE_D(d, list) GT_PP_LIST_FOLD_LEFT_ ## d(GT_PP_LIST_REVERSE_O, GT_PP_NIL, list)
# else
#    define GT_PP_LIST_REVERSE_D(d, list) GT_PP_LIST_REVERSE_D_I(d, list)
#    define GT_PP_LIST_REVERSE_D_I(d, list) GT_PP_LIST_FOLD_LEFT_ ## d(GT_PP_LIST_REVERSE_O, GT_PP_NIL, list)
# endif
#
# else
#
# include <gridtools/preprocessor/control/iif.hpp>
# include <gridtools/preprocessor/facilities/identity.hpp>
# include <gridtools/preprocessor/list/adt.hpp>
# include <gridtools/preprocessor/list/fold_left.hpp>
#
# /* GT_PP_LIST_REVERSE */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_LIST_REVERSE(list) GT_PP_IIF(GT_PP_LIST_IS_CONS(list),GT_PP_LIST_REVERSE_CONS,GT_PP_IDENTITY_N(GT_PP_NIL,1))(list)
# else
#    define GT_PP_LIST_REVERSE(list) GT_PP_LIST_REVERSE_I(list)
#    define GT_PP_LIST_REVERSE_I(list) GT_PP_IIF(GT_PP_LIST_IS_CONS(list),GT_PP_LIST_REVERSE_CONS,GT_PP_IDENTITY_N(GT_PP_NIL,1))(list)
# endif
#
# define GT_PP_LIST_REVERSE_O(d, s, x) (x, s)
#
# /* GT_PP_LIST_REVERSE_D */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_LIST_REVERSE_D(d, list) GT_PP_IIF(GT_PP_LIST_IS_CONS(list),GT_PP_LIST_REVERSE_CONS_D,GT_PP_IDENTITY_N(GT_PP_NIL,2))(d,list)
# else
#    define GT_PP_LIST_REVERSE_D(d, list) GT_PP_LIST_REVERSE_D_I(d, list)
#    define GT_PP_LIST_REVERSE_D_I(d, list) GT_PP_IIF(GT_PP_LIST_IS_CONS(list),GT_PP_LIST_REVERSE_CONS_D,GT_PP_IDENTITY_N(GT_PP_NIL,2))(d,list)
# endif
#
# define GT_PP_LIST_REVERSE_CONS(list) GT_PP_LIST_FOLD_LEFT(GT_PP_LIST_REVERSE_O, (GT_PP_LIST_FIRST(list),GT_PP_NIL), GT_PP_LIST_REST(list))
# define GT_PP_LIST_REVERSE_CONS_D(d, list) GT_PP_LIST_FOLD_LEFT_ ## d(GT_PP_LIST_REVERSE_O, (GT_PP_LIST_FIRST(list),GT_PP_NIL), GT_PP_LIST_REST(list))
#
# endif
#
# endif
