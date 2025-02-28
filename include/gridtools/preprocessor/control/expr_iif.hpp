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
# ifndef GT_PREPROCESSOR_CONTROL_EXPR_IIF_HPP
# define GT_PREPROCESSOR_CONTROL_EXPR_IIF_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# /* GT_PP_EXPR_IIF */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_EXPR_IIF(bit, expr) GT_PP_EXPR_IIF_I(bit, expr)
# else
#    define GT_PP_EXPR_IIF(bit, expr) GT_PP_EXPR_IIF_OO((bit, expr))
#    define GT_PP_EXPR_IIF_OO(par) GT_PP_EXPR_IIF_I ## par
# endif
#
# define GT_PP_EXPR_IIF_I(bit, expr) GT_PP_EXPR_IIF_ ## bit(expr)
#
# define GT_PP_EXPR_IIF_0(expr)
# define GT_PP_EXPR_IIF_1(expr) expr
#
# endif
