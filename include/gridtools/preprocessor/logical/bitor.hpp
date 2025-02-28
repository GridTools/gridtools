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
# ifndef GT_PREPROCESSOR_LOGICAL_BITOR_HPP
# define GT_PREPROCESSOR_LOGICAL_BITOR_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# /* GT_PP_BITOR */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_BITOR(x, y) GT_PP_BITOR_I(x, y)
# else
#    define GT_PP_BITOR(x, y) GT_PP_BITOR_OO((x, y))
#    define GT_PP_BITOR_OO(par) GT_PP_BITOR_I ## par
# endif
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MSVC()
#    define GT_PP_BITOR_I(x, y) GT_PP_BITOR_ ## x ## y
# else
#    define GT_PP_BITOR_I(x, y) GT_PP_BITOR_ID(GT_PP_BITOR_ ## x ## y)
#    define GT_PP_BITOR_ID(id) id
# endif
#
# define GT_PP_BITOR_00 0
# define GT_PP_BITOR_01 1
# define GT_PP_BITOR_10 1
# define GT_PP_BITOR_11 1
#
# endif
