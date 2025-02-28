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
# ifndef GT_PREPROCESSOR_ARRAY_DATA_HPP
# define GT_PREPROCESSOR_ARRAY_DATA_HPP
#
# include <gridtools/preprocessor/config/config.hpp>
# include <gridtools/preprocessor/tuple/elem.hpp>
#
# /* GT_PP_ARRAY_DATA */
#
# if ~GT_PP_CONFIG_FLAGS() & GT_PP_CONFIG_EDG()
#    define GT_PP_ARRAY_DATA(array) GT_PP_TUPLE_ELEM(2, 1, array)
# else
#    define GT_PP_ARRAY_DATA(array) GT_PP_ARRAY_DATA_I(array)
#    define GT_PP_ARRAY_DATA_I(array) GT_PP_ARRAY_DATA_II array
#    define GT_PP_ARRAY_DATA_II(size, data) data
# endif
#
# endif
