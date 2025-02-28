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
# ifndef GT_PREPROCESSOR_REPETITION_ENUM_PARAMS_HPP
# define GT_PREPROCESSOR_REPETITION_ENUM_PARAMS_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/punctuation/comma_if.hpp>
# include <gridtools/preprocessor/repetition/repeat.hpp>
#
# /* GT_PP_ENUM_PARAMS */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_ENUM_PARAMS(count, param) GT_PP_REPEAT(count, GT_PP_ENUM_PARAMS_M, param)
# else
#    define GT_PP_ENUM_PARAMS(count, param) GT_PP_ENUM_PARAMS_I(count, param)
#    define GT_PP_ENUM_PARAMS_I(count, param) GT_PP_REPEAT(count, GT_PP_ENUM_PARAMS_M, param)
# endif
#
# define GT_PP_ENUM_PARAMS_M(z, n, param) GT_PP_COMMA_IF(n) param ## n
#
# /* GT_PP_ENUM_PARAMS_Z */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_ENUM_PARAMS_Z(z, count, param) GT_PP_REPEAT_ ## z(count, GT_PP_ENUM_PARAMS_M, param)
# else
#    define GT_PP_ENUM_PARAMS_Z(z, count, param) GT_PP_ENUM_PARAMS_Z_I(z, count, param)
#    define GT_PP_ENUM_PARAMS_Z_I(z, count, param) GT_PP_REPEAT_ ## z(count, GT_PP_ENUM_PARAMS_M, param)
# endif
#
# endif
