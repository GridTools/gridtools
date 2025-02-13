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
# ifndef GT_PREPROCESSOR_COMPARISON_GREATER_HPP
# define GT_PREPROCESSOR_COMPARISON_GREATER_HPP
#
# include <gridtools/preprocessor/comparison/less.hpp>
# include <gridtools/preprocessor/config/config.hpp>
#
# /* GT_PP_GREATER */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_GREATER(x, y) GT_PP_LESS(y, x)
# else
#    define GT_PP_GREATER(x, y) GT_PP_GREATER_I(x, y)
#    define GT_PP_GREATER_I(x, y) GT_PP_LESS(y, x)
# endif
#
# /* GT_PP_GREATER_D */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_GREATER_D(d, x, y) GT_PP_LESS_D(d, y, x)
# else
#    define GT_PP_GREATER_D(d, x, y) GT_PP_GREATER_D_I(d, x, y)
#    define GT_PP_GREATER_D_I(d, x, y) GT_PP_LESS_D(d, y, x)
# endif
#
# endif
