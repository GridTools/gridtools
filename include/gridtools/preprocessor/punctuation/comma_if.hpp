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
# ifndef GT_PREPROCESSOR_PUNCTUATION_COMMA_IF_HPP
# define GT_PREPROCESSOR_PUNCTUATION_COMMA_IF_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/control/if.hpp>
# include <gridtools/preprocessor/facilities/empty.hpp>
# include <gridtools/preprocessor/punctuation/comma.hpp>
#
# /* GT_PP_COMMA_IF */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_COMMA_IF(cond) GT_PP_IF(cond, GT_PP_COMMA, GT_PP_EMPTY)()
# else
#    define GT_PP_COMMA_IF(cond) GT_PP_COMMA_IF_I(cond)
#    define GT_PP_COMMA_IF_I(cond) GT_PP_IF(cond, GT_PP_COMMA, GT_PP_EMPTY)()
# endif
#
# endif
