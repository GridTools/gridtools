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
# ifndef GT_PREPROCESSOR_ARRAY_SIZE_HPP
# define GT_PREPROCESSOR_ARRAY_SIZE_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/tuple/elem.hpp>
#
# /* GT_PP_ARRAY_SIZE */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_ARRAY_SIZE(array) GT_PP_TUPLE_ELEM(2, 0, array)
# else
#    define GT_PP_ARRAY_SIZE(array) GT_PP_ARRAY_SIZE_I(array)
#    define GT_PP_ARRAY_SIZE_I(array) GT_PP_ARRAY_SIZE_II array
#    define GT_PP_ARRAY_SIZE_II(size, data) size
# endif
#
# endif
