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
# ifndef GT_PREPROCESSOR_COMPARISON_LESS_HPP
# define GT_PREPROCESSOR_COMPARISON_LESS_HPP
#
# include <gridtools/preprocessor/comparison/less_equal.hpp>
# include <gridtools/preprocessor/comparison/not_equal.hpp>
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/control/iif.hpp>
# include <gridtools/preprocessor/logical/bitand.hpp>
# include <gridtools/preprocessor/tuple/eat.hpp>
#
# /* GT_PP_LESS */
#
# if GT_PP_CONFIG_FLAGS() & (GT_PP_CONFIG_MWCC() | GT_PP_CONFIG_DMC())
#    define GT_PP_LESS(x, y) GT_PP_BITAND(GT_PP_NOT_EQUAL(x, y), GT_PP_LESS_EQUAL(x, y))
# elif ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_LESS(x, y) GT_PP_IIF(GT_PP_NOT_EQUAL(x, y), GT_PP_LESS_EQUAL, 0 GT_PP_TUPLE_EAT_2)(x, y)
# else
#    define GT_PP_LESS(x, y) GT_PP_LESS_I(x, y)
#    define GT_PP_LESS_I(x, y) GT_PP_IIF(GT_PP_NOT_EQUAL(x, y), GT_PP_LESS_EQUAL, 0 GT_PP_TUPLE_EAT_2)(x, y)
# endif
#
# /* GT_PP_LESS_D */
#
# if GT_PP_CONFIG_FLAGS() & (GT_PP_CONFIG_MWCC() | GT_PP_CONFIG_DMC())
#    define GT_PP_LESS_D(d, x, y) GT_PP_BITAND(GT_PP_NOT_EQUAL(x, y), GT_PP_LESS_EQUAL_D(d, x, y))
# elif ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_LESS_D(d, x, y) GT_PP_IIF(GT_PP_NOT_EQUAL(x, y), GT_PP_LESS_EQUAL_D, 0 GT_PP_TUPLE_EAT_3)(d, x, y)
# else
#    define GT_PP_LESS_D(d, x, y) GT_PP_LESS_D_I(d, x, y)
#    define GT_PP_LESS_D_I(d, x, y) GT_PP_IIF(GT_PP_NOT_EQUAL(x, y), GT_PP_LESS_EQUAL_D, 0 GT_PP_TUPLE_EAT_3)(d, x, y)
# endif
#
# endif
