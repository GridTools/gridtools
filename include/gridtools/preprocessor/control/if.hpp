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
# ifndef GT_PREPROCESSOR_CONTROL_IF_HPP
# define GT_PREPROCESSOR_CONTROL_IF_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/control/iif.hpp>
# include <gridtools/preprocessor/logical/bool.hpp>
#
# /* GT_PP_IF */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_IF(cond, t, f) GT_PP_IIF(GT_PP_BOOL(cond), t, f)
# else
#    define GT_PP_IF(cond, t, f) GT_PP_IF_I(cond, t, f)
#    define GT_PP_IF_I(cond, t, f) GT_PP_IIF(GT_PP_BOOL(cond), t, f)
# endif
#
# endif
