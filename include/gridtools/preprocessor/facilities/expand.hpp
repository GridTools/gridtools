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
# ifndef GT_PREPROCESSOR_FACILITIES_EXPAND_HPP
# define GT_PREPROCESSOR_FACILITIES_EXPAND_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC() && ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_DMC()
#    define GT_PP_EXPAND(x) GT_PP_EXPAND_I(x)
# else
#    define GT_PP_EXPAND(x) GT_PP_EXPAND_OO((x))
#    define GT_PP_EXPAND_OO(par) GT_PP_EXPAND_I ## par
# endif
#
# define GT_PP_EXPAND_I(x) x
#
# endif
