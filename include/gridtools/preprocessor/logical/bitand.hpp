# /* **************************************************************************
#  *                                                                          *
#  *     (C) Copyright Paul Mensonides 2002.
#  *     Distributed under the Boost Software License, Version 1.0. (See
#  *     accompanying file LICENSE_1_0.txt or copy at
#  *     http://www.boost.org/LICENSE_1_0.txt)
#  *                                                                          *
#  ************************************************************************** */
#
# /* See http://www.boost.org for most recent version. */
#
# ifndef GT_PREPROCESSOR_LOGICAL_BITAND_HPP
# define GT_PREPROCESSOR_LOGICAL_BITAND_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# /* GT_PP_BITAND */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_BITAND(x, y) GT_PP_BITAND_I(x, y)
# else
#    define GT_PP_BITAND(x, y) GT_PP_BITAND_OO((x, y))
#    define GT_PP_BITAND_OO(par) GT_PP_BITAND_I ## par
# endif
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MSVC()
#    define GT_PP_BITAND_I(x, y) GT_PP_BITAND_ ## x ## y
# else
#    define GT_PP_BITAND_I(x, y) GT_PP_BITAND_ID(GT_PP_BITAND_ ## x ## y)
#    define GT_PP_BITAND_ID(res) res
# endif
#
# define GT_PP_BITAND_00 0
# define GT_PP_BITAND_01 0
# define GT_PP_BITAND_10 0
# define GT_PP_BITAND_11 1
#
# endif
