# /* Copyright (C) 2001
#  * Housemarque Oy
#  * http://www.housemarque.com
#  *
#  * Distributed under the Boost Software License, Version 1.0. (See
#  * accompanying file LICENSE_1_0.txt or copy at
#  * http://www.boost.org/LICENSE_1_0.txt)
#  *
#  * See http://www.boost.org for most recent version.
#  */
#
# /* Revised by Paul Mensonides (2002) */
#
# ifndef GT_PREPROCESSOR_LIST_ADT_HPP
# define GT_PREPROCESSOR_LIST_ADT_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/detail/is_binary.hpp>
# include <gridtools/preprocessor/logical/compl.hpp>
# include <gridtools/preprocessor/tuple/eat.hpp>
#
# /* GT_PP_LIST_CONS */
#
# define GT_PP_LIST_CONS(head, tail) (head, tail)
#
# /* GT_PP_LIST_NIL */
#
# define GT_PP_LIST_NIL GT_PP_NIL
#
# /* GT_PP_LIST_FIRST */
#
# define GT_PP_LIST_FIRST(list) GT_PP_LIST_FIRST_D(list)
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_LIST_FIRST_D(list) GT_PP_LIST_FIRST_I list
# else
#    define GT_PP_LIST_FIRST_D(list) GT_PP_LIST_FIRST_I ## list
# endif
#
# define GT_PP_LIST_FIRST_I(head, tail) head
#
# /* GT_PP_LIST_REST */
#
# define GT_PP_LIST_REST(list) GT_PP_LIST_REST_D(list)
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_LIST_REST_D(list) GT_PP_LIST_REST_I list
# else
#    define GT_PP_LIST_REST_D(list) GT_PP_LIST_REST_I ## list
# endif
#
# define GT_PP_LIST_REST_I(head, tail) tail
#
# /* GT_PP_LIST_IS_CONS */
#
# if GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_BCC()
#    define GT_PP_LIST_IS_CONS(list) GT_PP_LIST_IS_CONS_D(list)
#    define GT_PP_LIST_IS_CONS_D(list) GT_PP_LIST_IS_CONS_ ## list
#    define GT_PP_LIST_IS_CONS_(head, tail) 1
#    define GT_PP_LIST_IS_CONS_GT_PP_NIL 0
# else
#    define GT_PP_LIST_IS_CONS(list) GT_PP_IS_BINARY(list)
# endif
#
# /* GT_PP_LIST_IS_NIL */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_BCC()
#    define GT_PP_LIST_IS_NIL(list) GT_PP_COMPL(GT_PP_IS_BINARY(list))
# else
#    define GT_PP_LIST_IS_NIL(list) GT_PP_COMPL(GT_PP_LIST_IS_CONS(list))
# endif
#
# endif
