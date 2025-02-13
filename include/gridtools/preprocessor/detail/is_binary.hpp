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
# ifndef GT_PREPROCESSOR_DETAIL_IS_BINARY_HPP
# define GT_PREPROCESSOR_DETAIL_IS_BINARY_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/detail/check.hpp>
#
# /* GT_PP_IS_BINARY */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_IS_BINARY(x) GT_PP_CHECK(x, GT_PP_IS_BINARY_CHECK)
# else
#    define GT_PP_IS_BINARY(x) GT_PP_IS_BINARY_I(x)
#    define GT_PP_IS_BINARY_I(x) GT_PP_CHECK(x, GT_PP_IS_BINARY_CHECK)
# endif
#
# define GT_PP_IS_BINARY_CHECK(a, b) 1
# define GT_PP_CHECK_RESULT_GT_PP_IS_BINARY_CHECK 0, GT_PP_NIL
#
# endif
