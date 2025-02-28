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
# ifndef GT_PREPROCESSOR_COMPARISON_EQUAL_HPP
# define GT_PREPROCESSOR_COMPARISON_EQUAL_HPP
#
# include <gridtools/preprocessor/comparison/not_equal.hpp>
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/logical/compl.hpp>
#
# /* GT_PP_EQUAL */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_EQUAL(x, y) GT_PP_COMPL(GT_PP_NOT_EQUAL(x, y))
# else
#    define GT_PP_EQUAL(x, y) GT_PP_EQUAL_I(x, y)
#    define GT_PP_EQUAL_I(x, y) GT_PP_COMPL(GT_PP_NOT_EQUAL(x, y))
# endif
#
# /* GT_PP_EQUAL_D */
#
# define GT_PP_EQUAL_D(d, x, y) GT_PP_EQUAL(x, y)
#
# endif
