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
# ifndef GT_PREPROCESSOR_CONTROL_IIF_HPP
# define GT_PREPROCESSOR_CONTROL_IIF_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_IIF(bit, t, f) GT_PP_IIF_I(bit, t, f)
# else
#    define GT_PP_IIF(bit, t, f) GT_PP_IIF_OO((bit, t, f))
#    define GT_PP_IIF_OO(par) GT_PP_IIF_I ## par
# endif
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MSVC()
#    define GT_PP_IIF_I(bit, t, f) GT_PP_IIF_ ## bit(t, f)
# else
#    define GT_PP_IIF_I(bit, t, f) GT_PP_IIF_II(GT_PP_IIF_ ## bit(t, f))
#    define GT_PP_IIF_II(id) id
# endif
#
# define GT_PP_IIF_0(t, f) f
# define GT_PP_IIF_1(t, f) t
#
# endif
