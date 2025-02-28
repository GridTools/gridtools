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
# ifndef GT_PREPROCESSOR_STRINGIZE_HPP
# define GT_PREPROCESSOR_STRINGIZE_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
#
# /* GT_PP_STRINGIZE */
#
# if GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MSVC()
#    define GT_PP_STRINGIZE(text) GT_PP_STRINGIZE_A((text))
#    define GT_PP_STRINGIZE_A(arg) GT_PP_STRINGIZE_I arg
# elif GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_MWCC()
#    define GT_PP_STRINGIZE(text) GT_PP_STRINGIZE_OO((text))
#    define GT_PP_STRINGIZE_OO(par) GT_PP_STRINGIZE_I ## par
# else
#    define GT_PP_STRINGIZE(text) GT_PP_STRINGIZE_I(text)
# endif
#
# define GT_PP_STRINGIZE_I(...) #__VA_ARGS__
#
# endif
