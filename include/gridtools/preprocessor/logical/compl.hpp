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
# ifndef GT_PREPROCESSOR_LOGICAL_COMPL_HPP
# define GT_PREPROCESSOR_LOGICAL_COMPL_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# /* GT_PP_COMPL */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_COMPL(x) GT_PP_COMPL_I(x)
# else
#    define GT_PP_COMPL(x) GT_PP_COMPL_OO((x))
#    define GT_PP_COMPL_OO(par) GT_PP_COMPL_I ## par
# endif
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MSVC()
#    define GT_PP_COMPL_I(x) GT_PP_COMPL_ ## x
# else
#    define GT_PP_COMPL_I(x) GT_PP_COMPL_ID(GT_PP_COMPL_ ## x)
#    define GT_PP_COMPL_ID(id) id
# endif
#
# define GT_PP_COMPL_0 1
# define GT_PP_COMPL_1 0
#
# endif
