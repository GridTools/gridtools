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
# ifndef GT_PREPROCESSOR_CAT_HPP
# define GT_PREPROCESSOR_CAT_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# /* GT_PP_CAT */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_CAT(a, b) GT_PP_CAT_I(a, b)
# else
#    define GT_PP_CAT(a, b) GT_PP_CAT_OO((a, b))
#    define GT_PP_CAT_OO(par) GT_PP_CAT_I ## par
# endif
#
# if (~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MSVC()) || (defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 1700)
#    define GT_PP_CAT_I(a, b) a ## b
# else
#    define GT_PP_CAT_I(a, b) GT_PP_CAT_II(~, a ## b)
#    define GT_PP_CAT_II(p, res) res
# endif
#
# endif
