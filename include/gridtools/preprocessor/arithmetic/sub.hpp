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
# /* Revised by Edward Diener (2020) */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_ARITHMETIC_SUB_HPP
# define GT_PREPROCESSOR_ARITHMETIC_SUB_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_STRICT()
#
# include <gridtools/preprocessor/arithmetic/dec.hpp>
# include <gridtools/preprocessor/control/while.hpp>
# include <gridtools/preprocessor/tuple/elem.hpp>
#
# /* GT_PP_SUB */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_SUB(x, y) GT_PP_TUPLE_ELEM(2, 0, GT_PP_WHILE(GT_PP_SUB_P, GT_PP_SUB_O, (x, y)))
# else
#    define GT_PP_SUB(x, y) GT_PP_SUB_I(x, y)
#    define GT_PP_SUB_I(x, y) GT_PP_TUPLE_ELEM(2, 0, GT_PP_WHILE(GT_PP_SUB_P, GT_PP_SUB_O, (x, y)))
# endif
#
# define GT_PP_SUB_P(d, xy) GT_PP_TUPLE_ELEM(2, 1, xy)
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_SUB_O(d, xy) GT_PP_SUB_O_I xy
# else
#    define GT_PP_SUB_O(d, xy) GT_PP_SUB_O_I(GT_PP_TUPLE_ELEM(2, 0, xy), GT_PP_TUPLE_ELEM(2, 1, xy))
# endif
#
# define GT_PP_SUB_O_I(x, y) (GT_PP_DEC(x), GT_PP_DEC(y))
#
# /* GT_PP_SUB_D */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_SUB_D(d, x, y) GT_PP_TUPLE_ELEM(2, 0, GT_PP_WHILE_ ## d(GT_PP_SUB_P, GT_PP_SUB_O, (x, y)))
# else
#    define GT_PP_SUB_D(d, x, y) GT_PP_SUB_D_I(d, x, y)
#    define GT_PP_SUB_D_I(d, x, y) GT_PP_TUPLE_ELEM(2, 0, GT_PP_WHILE_ ## d(GT_PP_SUB_P, GT_PP_SUB_O, (x, y)))
# endif
#
# else
#
# include <gridtools/preprocessor/arithmetic/dec.hpp>
# include <gridtools/preprocessor/control/iif.hpp>
# include <gridtools/preprocessor/control/while.hpp>
# include <gridtools/preprocessor/facilities/identity.hpp>
# include <gridtools/preprocessor/logical/and.hpp>
# include <gridtools/preprocessor/logical/bitor.hpp>
# include <gridtools/preprocessor/tuple/elem.hpp>
# include <gridtools/preprocessor/arithmetic/detail/is_maximum_number.hpp>
# include <gridtools/preprocessor/arithmetic/detail/is_minimum_number.hpp>
#
# /* GT_PP_SUB */
#
#    define GT_PP_SUB(x, y) GT_PP_IIF(GT_PP_BITOR(GT_PP_DETAIL_IS_MAXIMUM_NUMBER(y),GT_PP_DETAIL_IS_MINIMUM_NUMBER(x)),GT_PP_IDENTITY_N(0,2),GT_PP_SUB_DO)(x,y)
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_SUB_DO(x, y) GT_PP_TUPLE_ELEM(2, 0, GT_PP_WHILE(GT_PP_SUB_P, GT_PP_SUB_O, (x, y)))
# else
#    define GT_PP_SUB_DO(x, y) GT_PP_SUB_I(x, y)
#    define GT_PP_SUB_I(x, y) GT_PP_TUPLE_ELEM(2, 0, GT_PP_WHILE(GT_PP_SUB_P, GT_PP_SUB_O, (x, y)))
# endif
#
# define GT_PP_SUB_P(d, xy) GT_PP_AND(GT_PP_TUPLE_ELEM(2, 1, xy),GT_PP_TUPLE_ELEM(2, 0, xy))
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_SUB_O(d, xy) GT_PP_SUB_O_I xy
# else
#    define GT_PP_SUB_O(d, xy) GT_PP_SUB_O_I(GT_PP_TUPLE_ELEM(2, 0, xy), GT_PP_TUPLE_ELEM(2, 1, xy))
# endif
#
# define GT_PP_SUB_O_I(x, y) (GT_PP_DEC(x), GT_PP_DEC(y))
#
# /* GT_PP_SUB_D */
#
#    define GT_PP_SUB_D(d, x, y) GT_PP_IIF(GT_PP_BITOR(GT_PP_DETAIL_IS_MAXIMUM_NUMBER(y),GT_PP_DETAIL_IS_MINIMUM_NUMBER(x)),GT_PP_IDENTITY_N(0,3),GT_PP_SUB_DO_D)(d,x,y)
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_SUB_DO_D(d, x, y) GT_PP_TUPLE_ELEM(2, 0, GT_PP_WHILE_ ## d(GT_PP_SUB_P, GT_PP_SUB_O, (x, y)))
# else
#    define GT_PP_SUB_DO_D(d, x, y) GT_PP_SUB_D_I(d, x, y)
#    define GT_PP_SUB_D_I(d, x, y) GT_PP_TUPLE_ELEM(2, 0, GT_PP_WHILE_ ## d(GT_PP_SUB_P, GT_PP_SUB_O, (x, y)))
# endif
#
# endif
#
# endif
