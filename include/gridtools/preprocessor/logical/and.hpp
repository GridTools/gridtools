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
# ifndef GT_PREPROCESSOR_LOGICAL_AND_HPP
# define GT_PREPROCESSOR_LOGICAL_AND_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/logical/bool.hpp>
# include <gridtools/preprocessor/logical/bitand.hpp>
#
# /* GT_PP_AND */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_AND(p, q) GT_PP_BITAND(GT_PP_BOOL(p), GT_PP_BOOL(q))
# else
#    define GT_PP_AND(p, q) GT_PP_AND_I(p, q)
#    define GT_PP_AND_I(p, q) GT_PP_BITAND(GT_PP_BOOL(p), GT_PP_BOOL(q))
# endif
#
# endif
